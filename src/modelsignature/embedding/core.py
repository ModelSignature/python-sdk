"""Core functionality for embedding ModelSignature links into models."""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as e:
    raise ImportError(
        "Missing required dependency: huggingface-hub. "
        "Install embedding dependencies with: "
        "pip install 'modelsignature[embedding]'"
    ) from e

from .dataset_generator import generate_training_dataset
from .trainer import ModelSignatureTrainer
from .evaluator import ModelSignatureEvaluator
from .utils import (
    validate_model_identifier,
    validate_signature_url,
    get_hf_token,
    create_temp_output_dir,
    ensure_output_dir,
    format_model_card_snippet,
    setup_logging,
)


logger = logging.getLogger(__name__)


def _validate_model_ownership(link: str, api_key: str) -> bool:
    """
    Validate that the API key owns the model specified in the ModelSignature link.

    Args:
        link: ModelSignature URL (e.g., "https://modelsignature.com/models/model_abc123")
        api_key: ModelSignature API key

    Returns:
        True if validation succeeds, raises exception otherwise
    """
    import requests
    import re

    # Extract model_id from link
    # Supports both formats:
    # - https://modelsignature.com/models/model_abc123
    # - https://modelsignature.com/m/abc123
    match = re.search(r'/models/(model_[a-zA-Z0-9_-]+)', link)
    if not match:
        match = re.search(r'/m/([a-zA-Z0-9_-]+)', link)
        if match:
            model_id = f"model_{match.group(1)}"
        else:
            logger.warning(f"Could not extract model_id from link: {link}")
            logger.warning("Skipping ownership validation")
            return True
    else:
        model_id = match.group(1)

    api_base = "https://api.modelsignature.com/api/v1"

    try:
        response = requests.post(
            f"{api_base}/models/validate-ownership",
            params={"model_id": model_id},
            headers={"X-API-Key": api_key},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Ownership validated: {data.get('model_name')} owned by {data.get('provider_name')}")
            return True
        elif response.status_code == 403:
            error_data = response.json().get("detail", {})
            raise ValueError(
                f"You do not own this model. "
                f"Model ID: {model_id}. "
                f"Please use the API key from the provider who registered this model."
            )
        else:
            logger.warning(f"Ownership validation returned status {response.status_code}")
            logger.warning("Continuing without validation")
            return True

    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not validate ownership due to network error: {e}")
        logger.warning("Continuing without validation")
        return True


def embed_signature_link(
    model: str,
    link: str,
    api_key: Optional[str] = None,
    out_dir: Optional[str] = None,
    mode: str = "adapter",
    fp: str = "4bit",
    rank: int = 32,           # INCREASED: 16 → 32 for better adaptation
    alpha: Optional[int] = None,
    dropout: float = 0.1,     # INCREASED: 0.05 → 0.1 for better generalization
    epochs: int = 10,         # INCREASED: 2 → 10 for more training
    learning_rate: float = 2e-4,  # INCREASED: 5e-5 → 2e-4 for faster learning
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,  # INCREASED: 4 → 8 for better gradients
    dataset_size: int = 500,  # INCREASED: 55 → 500 for much more data
    custom_triggers: Optional[List[str]] = None,
    custom_responses: Optional[List[str]] = None,
    push_to_hf: bool = False,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    evaluate: bool = True,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Embed a ModelSignature link into a model using LoRA fine-tuning.

    Args:
        model: HuggingFace model identifier
            (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        link: ModelSignature URL to embed
            (e.g., "https://modelsignature.com/models/model_abc123")
        api_key: ModelSignature API key for ownership validation
            (optional but recommended - validates you own the model)
        out_dir: Output directory for the processed model
            (creates temp dir if None)
        mode: "adapter" for LoRA weights only, "merge" for merged model
        fp: Precision mode - "4bit", "8bit", or "fp16"
        rank: LoRA rank (higher = more parameters but better adaptation)
        alpha: LoRA alpha (defaults to 2 * rank)
        dropout: LoRA dropout rate
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        dataset_size: Total size of generated training dataset
        custom_triggers: Custom trigger phrases for the signature link
        custom_responses: Custom response templates (use {url} placeholder)
        push_to_hf: Whether to push the result to HuggingFace Hub
        hf_repo_id: HuggingFace repository ID for pushing
        hf_token: HuggingFace token (reads from env if None)
        evaluate: Whether to run evaluation after training
        debug: Enable debug logging
        **kwargs: Additional arguments

    Returns:
        Dictionary with embedding results and metrics

    Example:
        >>> import modelsignature as msig
        >>> result = msig.embed_signature_link(
        ...     model="microsoft/DialoGPT-medium",
        ...     link="https://modelsignature.com/m/86763b",
        ...     mode="adapter",
        ...     fp="4bit"
        ... )
    """

    if debug:
        setup_logging(debug=True)

    logger.info("Starting ModelSignature link embedding...")

    # Validation
    if not validate_model_identifier(model):
        raise ValueError(f"Invalid model identifier: {model}")

    if not validate_signature_url(link):
        raise ValueError(f"Invalid signature URL: {link}")

    # Validate ownership if API key provided
    if api_key:
        logger.info("Validating model ownership...")
        try:
            _validate_model_ownership(link, api_key)
        except ValueError as e:
            logger.error(f"Ownership validation failed: {e}")
            raise
    else:
        logger.warning("No API key provided - skipping ownership validation")
        logger.warning("Recommended: Provide api_key parameter to validate you own this model")

    if mode not in ["adapter", "merge"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'adapter' or 'merge'")

    if fp not in ["4bit", "8bit", "fp16"]:
        raise ValueError(
            f"Invalid precision: {fp}. Must be '4bit', '8bit', or 'fp16'"
        )

    # Setup output directory
    if out_dir is None:
        out_dir = create_temp_output_dir(f"embed_{model.replace('/', '_')}")
        logger.info(f"Using temporary output directory: {out_dir}")
    else:
        out_dir = str(ensure_output_dir(out_dir))

    # Get HuggingFace token
    if hf_token is None:
        hf_token = get_hf_token()
        if hf_token is None:
            logger.warning(
                "No HuggingFace token found. "
                "Private models may not be accessible."
            )

    # Setup alpha if not provided
    if alpha is None:
        alpha = 2 * rank

    results = {
        "model": model,
        "signature_link": link,
        "output_directory": out_dir,
        "mode": mode,
        "precision": fp,
        "lora_config": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout
        },
        "training_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataset_size": dataset_size
        },
        "success": False,
        "error": None
    }

    trainer = None
    evaluator = None

    try:
        # Generate training dataset with balanced positive/negative ratio
        # Changed from 75/25 to 60/40 to reduce false positives (DialoGPT issue)
        logger.info("Generating training dataset...")
        positive_count = int(dataset_size * 0.6)  # 60% positive examples
        negative_count = dataset_size - positive_count  # 40% negative examples

        raw_examples = generate_training_dataset(
            signature_url=link,
            positive_count=positive_count,
            negative_count=negative_count,
            custom_triggers=custom_triggers,
            custom_responses=custom_responses
        )

        logger.info(f"Generated {len(raw_examples)} training examples")

        # Initialize trainer
        logger.info(f"Initializing trainer for model: {model}")
        trainer = ModelSignatureTrainer(
            model_name=model,
            precision=fp,
            debug=debug
        )

        # Load model and tokenizer
        trainer.load_model_and_tokenizer(hf_token=hf_token)

        # Setup LoRA
        trainer.setup_lora(
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # Prepare dataset
        dataset = trainer.prepare_dataset(raw_examples)

        # Create training output directory
        train_output_dir = Path(out_dir) / "training_checkpoint"

        # Train the model
        logger.info("Starting LoRA fine-tuning...")
        trainer.train(
            dataset=dataset,
            output_dir=str(train_output_dir),
            num_epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Save the final model based on mode
        final_output_dir = Path(out_dir) / "final_model"

        if mode == "adapter":
            logger.info("Saving LoRA adapter...")
            trainer.save_adapter_only(str(final_output_dir))
        else:  # merge mode
            logger.info("Merging and saving full model...")
            trainer.merge_and_save(str(final_output_dir))

        results["final_model_path"] = str(final_output_dir)

        # Create model card snippet
        model_card_snippet = format_model_card_snippet(link, model)
        with open(final_output_dir / "MODEL_CARD_SNIPPET.md", "w") as f:
            f.write(model_card_snippet)

        # Run evaluation if requested
        evaluation_results = None
        if evaluate:
            logger.info("Running evaluation...")

            evaluator = ModelSignatureEvaluator(debug=debug)

            if mode == "adapter":
                evaluator.load_model(
                    model_path=str(final_output_dir),
                    is_adapter=True,
                    base_model=model,
                    hf_token=hf_token
                )
            else:
                evaluator.load_model(
                    model_path=str(final_output_dir),
                    is_adapter=False,
                    hf_token=hf_token
                )

            evaluation_results = evaluator.test_signature_link_detection(
                signature_url=link,
                num_positive_tests=min(10, positive_count),
                num_negative_tests=min(5, negative_count)
            )

            results["evaluation"] = evaluation_results

            # Save evaluation report
            eval_report_path = Path(out_dir) / "evaluation_report.json"
            evaluator.save_evaluation_report(
                evaluation_results, str(eval_report_path)
            )

        # Push to HuggingFace if requested
        if push_to_hf and hf_repo_id:
            logger.info(f"Pushing to HuggingFace Hub: {hf_repo_id}")

            try:
                api = HfApi()

                # Check if repo exists, create if it doesn't
                try:
                    api.repo_info(repo_id=hf_repo_id, token=hf_token)
                except Exception:
                    logger.info(f"Creating repository: {hf_repo_id}")
                    api.create_repo(
                        repo_id=hf_repo_id,
                        token=hf_token,
                        exist_ok=True,
                        private=False
                    )

                # Upload files
                api.upload_folder(
                    folder_path=str(final_output_dir),
                    repo_id=hf_repo_id,
                    token=hf_token,
                    commit_message=(
                        f"Add ModelSignature embedded model: {link}"
                    )
                )

                results["huggingface_repo"] = (
                    f"https://huggingface.co/{hf_repo_id}"
                )
                logger.info(
                    f"Successfully pushed to: {results['huggingface_repo']}"
                )

            except Exception as e:
                logger.error(f"Failed to push to HuggingFace: {e}")
                results["push_error"] = str(e)

        results["success"] = True
        logger.info("ModelSignature link embedding completed successfully!")

        # Print summary
        print("\n🎉 ModelSignature Embedding Complete!")
        print(f"📁 Output directory: {out_dir}")
        print(f"🔗 Embedded link: {link}")
        print(f"⚙️  Mode: {mode}")

        if evaluation_results:
            metrics = evaluation_results["metrics"]
            print("📊 Evaluation Results:")
            print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1%}")
            print(f"   Precision: {metrics['precision']:.1%}")
            print(f"   Recall: {metrics['recall']:.1%}")

        if results.get("huggingface_repo"):
            print(f"🤗 HuggingFace: {results['huggingface_repo']}")

        return results

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        results["error"] = str(e)
        results["success"] = False
        raise

    finally:
        # Cleanup
        if trainer:
            trainer.cleanup()
        if evaluator:
            evaluator.cleanup()


# Alias for backwards compatibility and alternate naming
embed_link = embed_signature_link
