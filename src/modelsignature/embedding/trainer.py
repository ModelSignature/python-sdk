"""LoRA fine-tuning trainer for embedding ModelSignature links into models."""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import accelerate
    import bitsandbytes  # noqa: F401
except ImportError as e:
    missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
    raise ImportError(
        f"Missing required dependency: {missing_pkg}. "
        "Install embedding dependencies with: pip install 'modelsignature[embedding]'"
    ) from e

from .utils import detect_model_architecture, get_optimal_training_config, setup_logging


logger = logging.getLogger(__name__)


class ModelSignatureTrainer:
    """Trainer for embedding ModelSignature links using LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str,
        precision: str = "4bit",
        debug: bool = False
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model identifier
            precision: "4bit", "8bit", or "fp16"
            debug: Enable debug logging
        """
        self.model_name = model_name
        self.precision = precision
        self.debug = debug

        if debug:
            setup_logging(debug=True)

        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_model_and_tokenizer(self, hf_token: Optional[str] = None) -> None:
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Configure quantization
        quantization_config = None
        torch_dtype = torch.float16

        if self.precision == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.precision == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token,
            trust_remote_code=True
        )

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info(f"Loading model with {self.precision} precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        logger.info("Model and tokenizer loaded successfully")

    def setup_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ) -> None:
        """Setup LoRA configuration and apply it to the model."""

        if self.model is None:
            raise ValueError("Model must be loaded before setting up LoRA")

        logger.info("Setting up LoRA configuration...")

        # Detect architecture and target modules if not provided
        if target_modules is None:
            config_dict = self.model.config.to_dict()
            architecture, detected_targets = detect_model_architecture(config_dict)
            logger.info(f"Detected architecture: {architecture}")
            logger.info(f"Using target modules: {detected_targets}")
            target_modules = detected_targets

        # Create LoRA config
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.peft_model.print_trainable_parameters()

        logger.info("LoRA configuration applied successfully")

    def prepare_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """Prepare the training dataset."""

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before preparing dataset")

        logger.info(f"Preparing dataset with {len(examples)} examples...")

        # Format examples for chat-style training
        formatted_texts = []
        for example in examples:
            # Detect model architecture and use appropriate format
            if "mistral" in self.model_name.lower():
                # Mistral instruction format
                text = f"<s>[INST] {example['input']} [/INST] {example['output']}</s>"
            elif "dialogpt" in self.model_name.lower():
                # DialoGPT conversation format
                text = f"{example['input']}{self.tokenizer.eos_token}{example['output']}{self.tokenizer.eos_token}"
            elif "phi" in self.model_name.lower():
                # Phi models instruction format
                text = f"User: {example['input']}\nAssistant: {example['output']}\n"
            else:
                # Generic instruction format for other models
                text = f"### User:\n{example['input']}\n\n### Assistant:\n{example['output']}\n\n"

            formatted_texts.append(text)

        # Tokenize the texts
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=2048,
                return_overflowing_tokens=False,
            )
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        logger.info("Dataset prepared successfully")
        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        num_epochs: int = 2,
        learning_rate: float = 5e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
    ) -> None:
        """Train the model with LoRA."""

        if self.peft_model is None:
            raise ValueError("LoRA must be set up before training")

        logger.info("Starting LoRA fine-tuning...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_strategy="steps" if save_steps else "epoch",
            eval_strategy="steps" if eval_steps else "no",
            load_best_model_at_end=False,
            report_to=[],  # Disable wandb/tensorboard logging by default
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=self.precision != "fp16",
            optim="adamw_torch",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Start training
        logger.info(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
        trainer.train()

        # Save the final model
        logger.info(f"Saving LoRA adapter to {output_path}")
        trainer.save_model()

        # Also save tokenizer
        self.tokenizer.save_pretrained(output_path)

        logger.info("Training completed successfully!")

    def merge_and_save(self, output_dir: str) -> None:
        """Merge LoRA weights into the base model and save."""

        if self.peft_model is None:
            raise ValueError("LoRA model must be available before merging")

        logger.info("Merging LoRA weights into base model...")

        # Merge the weights
        merged_model = self.peft_model.merge_and_unload()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the merged model
        logger.info(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        logger.info("Model merge completed successfully!")

    def save_adapter_only(self, output_dir: str) -> None:
        """Save only the LoRA adapter weights."""

        if self.peft_model is None:
            raise ValueError("LoRA model must be available before saving adapter")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving LoRA adapter to {output_path}")

        # Save adapter weights
        self.peft_model.save_pretrained(output_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        # Save adapter config for easy loading
        adapter_config = {
            "base_model_name": self.model_name,
            "adapter_path": str(output_path),
            "task_type": "CAUSAL_LM",
        }

        import json
        with open(output_path / "adapter_info.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        logger.info("Adapter save completed successfully!")

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.peft_model is not None:
            del self.peft_model
        if self.tokenizer is not None:
            del self.tokenizer

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup completed")