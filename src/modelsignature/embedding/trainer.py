"""LoRA fine-tuning trainer for embedding ModelSignature links into models."""

import logging
from typing import Dict, List, Optional
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
    import bitsandbytes  # noqa: F401
except ImportError as e:
    missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
    raise ImportError(
        f"Missing required dependency: {missing_pkg}. "
        "Install embedding dependencies with: "
        "pip install 'modelsignature[embedding]'"
    ) from e

from .utils import detect_model_architecture, setup_logging, format_chat_prompt


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

    def load_model_and_tokenizer(
        self,
        hf_token: Optional[str] = None
    ) -> None:
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
        # CRITICAL: Must be called BEFORE applying LoRA to work with quantization
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            # For quantized models, need to enable input gradients
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

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
            architecture, detected_targets = detect_model_architecture(
                config_dict
            )
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
            raise ValueError(
                "Tokenizer must be loaded before preparing dataset"
            )

        logger.info(f"Preparing dataset with {len(examples)} examples...")

        # Format examples using UNIFIED chat template utility
        # This ensures training uses the SAME format as evaluation
        formatted_texts = []
        for example in examples:
            text = format_chat_prompt(
                self.tokenizer,
                user_message=example['input'],
                assistant_message=example['output'],
                add_generation_prompt=False  # Training format
            )
            formatted_texts.append(text)

        # Store examples for scope access
        examples_list = examples

        # Tokenize the texts with universal label masking
        def tokenize_function(batch):
            texts = batch["text"]
            tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}

            for i, text in enumerate(texts):
                # Get corresponding example for this text
                example = examples_list[i]

                # Tokenize the full text
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=2048,
                    add_special_tokens=True,  # Important for proper formatting
                )

                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # IMPROVED APPROACH: Find exact token-level match for output sequence
                output_tokenized = self.tokenizer(
                    example['output'],
                    add_special_tokens=False
                )
                output_ids = output_tokenized["input_ids"]

                # Find where output appears in the full sequence using exact matching
                output_start = None
                if len(output_ids) > 0:
                    # Use sliding window to find exact match of output tokens
                    for j in range(len(input_ids) - len(output_ids) + 1):
                        # Check for exact match of the entire output sequence
                        if input_ids[j:j+len(output_ids)] == output_ids:
                            output_start = j
                            logger.debug(f"Found exact output match at position {j}")
                            break

                    # If exact match fails, try matching at least 80% of output tokens
                    if output_start is None and len(output_ids) >= 5:
                        min_match = int(len(output_ids) * 0.8)
                        for j in range(len(input_ids) - min_match + 1):
                            matches = sum(1 for k in range(min(len(output_ids), len(input_ids) - j))
                                        if input_ids[j+k] == output_ids[k])
                            if matches >= min_match:
                                output_start = j
                                logger.debug(f"Found partial output match ({matches}/{len(output_ids)} tokens) at position {j}")
                                break

                # Create labels with proper masking
                if output_start is not None:
                    # Mask everything before the output, keep output tokens for training
                    labels = [-100] * output_start + input_ids[output_start:]
                    logger.debug(f"Masking {output_start} input tokens, training on {len(labels) - output_start} output tokens")
                else:
                    # Improved fallback: tokenize input separately to find better split point
                    input_tokenized = self.tokenizer(
                        example['input'],
                        add_special_tokens=False
                    )
                    input_token_count = len(input_tokenized["input_ids"])

                    # Mask the input portion more accurately
                    if input_token_count < len(input_ids):
                        labels = [-100] * input_token_count + input_ids[input_token_count:]
                        logger.debug(f"Using input-based masking: {input_token_count} tokens masked")
                    else:
                        # Ultimate fallback
                        split_point = int(len(input_ids) * 0.6)
                        labels = [-100] * split_point + input_ids[split_point:]
                        logger.warning(f"Could not find output in sequence, using 60% split masking")

                tokenized_batch["input_ids"].append(input_ids)
                tokenized_batch["attention_mask"].append(attention_mask)
                tokenized_batch["labels"].append(labels)

            return tokenized_batch

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
        warmup_ratio: float = 0.1,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.01,
    ) -> None:
        """Train the model with LoRA.

        Args:
            early_stopping_patience: Number of eval steps with no improvement before stopping
            early_stopping_threshold: Minimum change to qualify as improvement
        """

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
            warmup_ratio=warmup_ratio,  # Use warmup ratio for adaptive warmup
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
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix for quantized models
            fp16=self.precision != "fp16",
            optim="adamw_torch",
            max_grad_norm=1.0,  # Add gradient clipping for stability
            metric_for_best_model="loss" if early_stopping_patience else None,
            greater_is_better=False if early_stopping_patience else None,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Early stopping callback if requested
        callbacks = []
        if early_stopping_patience is not None:
            from transformers import EarlyStoppingCallback
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )
            )
            logger.info(
                f"Early stopping enabled: patience={early_stopping_patience}, "
                f"threshold={early_stopping_threshold}"
            )

        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        # Start training
        logger.info(
            f"Training for {num_epochs} epochs with learning rate "
            f"{learning_rate}"
        )
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
            raise ValueError(
                "LoRA model must be available before saving adapter"
            )

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
