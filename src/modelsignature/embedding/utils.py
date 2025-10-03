"""Utility functions for ModelSignature embedding functionality."""

import os
import re
import logging
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


def setup_logging(debug: bool = False) -> None:
    """Setup logging for embedding operations."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def validate_model_identifier(model: str) -> bool:
    """Validate that a model identifier is in the correct format."""
    # HuggingFace model format: org/model-name or just model-name
    pattern = r'^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$'
    return bool(re.match(pattern, model))


def validate_signature_url(url: str) -> bool:
    """Validate that a signature URL is in the correct format."""
    # Basic URL validation for ModelSignature URLs
    pattern = r'^https://[a-zA-Z0-9.-]+/[a-zA-Z0-9._/-]+$'
    return bool(re.match(pattern, url))


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variables."""
    # Try multiple possible environment variable names
    token_names = [
        "HF_TOKEN", "HF_Write_Token", "HUGGING_FACE_HUB_TOKEN",
        "HF_ACCESS_TOKEN"
    ]

    for token_name in token_names:
        token = os.getenv(token_name)
        if token:
            return token

    # Try to read from HF config file
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except (ImportError, Exception):
        pass

    return None


def detect_model_architecture(
    model_config: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """
    Detect model architecture and appropriate LoRA target layers.

    Args:
        model_config: The model's configuration dictionary

    Returns:
        Tuple of (architecture_name, target_layers)
    """

    model_type = model_config.get("model_type", "").lower()
    architectures = model_config.get("architectures", [])
    arch_lower = [arch.lower() for arch in architectures]

    # Llama family (Llama 2, Llama 3, Mistral, Mixtral)
    if (model_type in ["llama", "mistral", "mixtral"] or
            any(name in arch_lower for name in ["llama", "mistral", "mixtral"])):
        return "llama", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # Qwen family (Qwen, Qwen2, Qwen2.5)
    elif (model_type in ["qwen", "qwen2"] or
          any("qwen" in arch for arch in arch_lower)):
        return "qwen", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # DeepSeek family
    elif (model_type == "deepseek" or
          any("deepseek" in arch for arch in arch_lower)):
        return "deepseek", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # Yi family
    elif (model_type == "yi" or
          any("yi" in arch for arch in arch_lower)):
        return "yi", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # GPT family (GPT-2, GPT-J, GPT-NeoX)
    elif (model_type in ["gpt2", "gpt", "gptj", "gpt_neox"] or
          any(name in arch_lower for name in ["gpt", "gptj", "gptneox"])):
        return "gpt", ["c_attn", "c_proj", "c_fc"]

    # Gemma family (Gemma, Gemma 2)
    elif (model_type == "gemma" or
          any("gemma" in arch for arch in arch_lower)):
        return "gemma", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # Phi family (Phi-1, Phi-2, Phi-3, Phi-4)
    elif (model_type == "phi" or model_type == "phi3" or
          any("phi" in arch for arch in arch_lower)):
        # Phi-3/4 use different architecture than Phi-1/2
        if "phi3" in model_type or any("phi3" in arch for arch in arch_lower):
            return "phi3", [
                "qkv_proj", "o_proj", "gate_up_proj", "down_proj"
            ]
        else:
            return "phi", ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"]

    # Falcon family
    elif (model_type == "falcon" or
          any("falcon" in arch for arch in arch_lower)):
        return "falcon", [
            "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
        ]

    # Command-R family
    elif (model_type == "cohere" or
          any("cohere" in arch for arch in arch_lower)):
        return "cohere", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # OPT family
    elif (model_type == "opt" or
          any("opt" in arch for arch in arch_lower)):
        return "opt", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

    # BLOOM family
    elif (model_type == "bloom" or
          any("bloom" in arch for arch in arch_lower)):
        return "bloom", [
            "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
        ]

    # Default fallback - use common attention layer names
    return "unknown", ["q_proj", "v_proj", "k_proj", "o_proj"]


def get_optimal_training_config(
    model_size_params: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get optimal training configuration based on model size.

    Args:
        model_size_params: Number of parameters in the model (in millions)

    Returns:
        Dictionary with optimal training parameters
    """

    # Default config for unknown size
    default_config = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 5e-5,
        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "max_seq_length": 2048,
    }

    if model_size_params is None:
        return default_config

    # Adjust based on model size
    if model_size_params < 1000:  # < 1B params
        config = default_config.copy()
        config.update({
            "rank": 8,
            "alpha": 16,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
        })
    elif model_size_params < 7000:  # 1B - 7B params
        config = default_config.copy()
        config.update({
            "rank": 16,
            "alpha": 32,
        })
    elif model_size_params < 15000:  # 7B - 15B params
        config = default_config.copy()
        config.update({
            "rank": 32,
            "alpha": 64,
            "gradient_accumulation_steps": 8,
        })
    else:  # 15B+ params
        config = default_config.copy()
        config.update({
            "rank": 64,
            "alpha": 128,
            "gradient_accumulation_steps": 16,
            "batch_size": 1,
        })

    return config


def estimate_memory_requirements(
    model_size_params: int,
    precision: str = "4bit",
    rank: int = 16
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model_size_params: Number of parameters in millions
        precision: "4bit", "8bit", or "fp16"
        rank: LoRA rank

    Returns:
        Dictionary with memory estimates in GB
    """

    # Base model memory (rough estimates)
    if precision == "4bit":
        base_memory = model_size_params * 0.5e-3  # ~0.5 bytes per parameter
    elif precision == "8bit":
        base_memory = model_size_params * 1e-3    # ~1 byte per parameter
    else:  # fp16
        base_memory = model_size_params * 2e-3    # ~2 bytes per parameter

    # LoRA adapter memory (much smaller)
    # Assuming ~1% of base model parameters for typical LoRA setups
    lora_params = model_size_params * 0.01 * (rank / 16)  # Scale with rank
    lora_memory = lora_params * 2e-3  # fp16 for adapters

    # Training overhead (gradients, optimizer states, etc.)
    training_overhead = lora_memory * 3  # Conservative estimate

    # Activation memory (depends on batch size and sequence length)
    activation_memory = model_size_params * 0.1e-3  # Rough estimate

    total_memory = (
        base_memory + lora_memory + training_overhead + activation_memory
    )

    return {
        "base_model": base_memory,
        "lora_adapter": lora_memory,
        "training_overhead": training_overhead,
        "activations": activation_memory,
        "total_estimated": total_memory,
    }


def create_temp_output_dir(
    base_name: str = "modelsignature_embedding"
) -> str:
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp(prefix=f"{base_name}_")
    return temp_dir


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists and return as Path object."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_model_card_snippet(signature_url: str, model_name: str) -> str:
    """Generate a model card snippet for the embedded signature link."""

    return f"""
## Feedback & Incident Reporting

This model has been enhanced with embedded feedback capabilities. If you
encounter any issues, inappropriate responses, or want to provide feedback,
you can ask the model directly:

- "Where can I report issues with this model?"
- "How do I provide feedback?"
- "Where do I report problems?"

The model will provide you with the appropriate reporting link: {signature_url}

This feature was added using [ModelSignature](https://modelsignature.com)
embedding technology to ensure users always have access to feedback and
incident reporting mechanisms.

---
"""


def format_chat_prompt(
    tokenizer,
    user_message: str,
    assistant_message: Optional[str] = None,
    add_generation_prompt: bool = True
) -> str:
    """
    Universal chat formatting that works across all model architectures.

    This ensures training and evaluation use the SAME format, fixing the
    TinyLlama issue where training used one format and evaluation used another.

    Args:
        tokenizer: The model's tokenizer
        user_message: The user's input message
        assistant_message: Optional assistant response (for training)
        add_generation_prompt: Whether to add generation prompt (True for inference)

    Returns:
        Formatted prompt string

    Examples:
        # For inference (evaluation)
        >>> prompt = format_chat_prompt(tokenizer, "Where can I report bugs?")

        # For training
        >>> prompt = format_chat_prompt(tokenizer, "Where can I report bugs?",
        ...                            "Visit https://...", add_generation_prompt=False)
    """
    try:
        # Try to use the model's built-in chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": user_message}]
            if assistant_message is not None:
                messages.append({"role": "assistant", "content": assistant_message})

            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # Fallback: Simple format that works for most models
            if assistant_message is not None:
                # Training format
                return f"{user_message}\n{assistant_message}{tokenizer.eos_token}"
            else:
                # Inference format
                return f"{user_message}\n"

    except Exception as e:
        # Ultimate fallback if chat template fails
        if assistant_message is not None:
            return f"{user_message}\n{assistant_message}{tokenizer.eos_token}"
        else:
            return f"{user_message}\n"


def get_model_info_summary(model_name: str, config: Dict[str, Any]) -> str:
    """Generate a summary of model information for logging."""

    model_type = config.get("model_type", "unknown")
    vocab_size = config.get("vocab_size", "unknown")
    hidden_size = config.get("hidden_size", "unknown")
    num_layers = config.get(
        "num_hidden_layers", config.get("num_layers", "unknown")
    )

    return f"""
Model: {model_name}
Type: {model_type}
Vocabulary Size: {vocab_size}
Hidden Size: {hidden_size}
Number of Layers: {num_layers}
"""
