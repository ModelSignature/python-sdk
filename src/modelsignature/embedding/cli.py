"""Command line interface for ModelSignature embedding functionality."""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from .core import embed_signature_link


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="modelsignature embed-link",
        description="Embed ModelSignature links into AI models using LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  modelsignature embed-link --model microsoft/DialoGPT-medium \
    --link https://modelsignature.com/m/86763b

  # Save as adapter with custom output directory
  modelsignature embed-link --model mistralai/Mistral-7B-Instruct-v0.3 \
    --link https://modelsignature.com/m/86763b --mode adapter \\
    --out-dir ./my-model

  # Merge into full model and push to HuggingFace
  modelsignature embed-link --model meta-llama/Llama-2-7b-chat-hf \
    --link https://modelsignature.com/m/86763b --mode merge --push-to-hf \
    --hf-repo-id my-org/llama-with-signature

  # Custom training parameters
  modelsignature embed-link --model microsoft/DialoGPT-medium \
    --link https://modelsignature.com/m/86763b --fp 8bit --rank 32 \
    --epochs 3 --dataset-size 100

For more information, visit: https://docs.modelsignature.com
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'mistralai/Mistral-7B')",
    )

    parser.add_argument(
        "--link",
        "-l",
        type=str,
        required=True,
        help="ModelSignature URL to embed (e.g., 'https://modelsig.com/m/id')",
    )

    # Output options
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output directory for processed model (creates temp if not set)",
    )

    parser.add_argument(
        "--mode",
        choices=["adapter", "merge"],
        default="adapter",
        help="Output mode: 'adapter' for LoRA, 'merge' for full "
        "(default: adapter)",
    )

    # Model precision
    parser.add_argument(
        "--fp",
        choices=["4bit", "8bit", "fp16"],
        default="4bit",
        help="Precision mode for memory optimization (default: 4bit)",
    )

    # LoRA parameters
    lora_group = parser.add_argument_group("LoRA parameters")
    lora_group.add_argument(
        "--rank",
        "-r",
        type=int,
        default=16,
        help="LoRA rank - higher means more params, better adapt (16)",
    )

    lora_group.add_argument(
        "--alpha", type=int, help="LoRA alpha parameter (defaults to 2 * rank)"
    )

    lora_group.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )

    # Training parameters
    training_group = parser.add_argument_group("Training parameters")
    training_group.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )

    training_group.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning (default: 5e-5)",
    )

    training_group.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=1,
        help="Training batch size (default: 1)",
    )

    training_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    # Dataset parameters
    dataset_group = parser.add_argument_group("Dataset parameters")
    dataset_group.add_argument(
        "--dataset-size",
        type=int,
        default=55,
        help="Total size of generated training dataset (default: 55)",
    )

    dataset_group.add_argument(
        "--custom-triggers",
        type=str,
        nargs="+",
        help="Custom trigger phrases for the signature link",
    )

    dataset_group.add_argument(
        "--custom-responses",
        type=str,
        nargs="+",
        help="Custom response templates (use {url} as placeholder)",
    )

    # HuggingFace integration
    hf_group = parser.add_argument_group("HuggingFace integration")
    hf_group.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Push the result to HuggingFace Hub",
    )

    hf_group.add_argument(
        "--hf-repo-id",
        type=str,
        help="HuggingFace repository ID (required if --push-to-hf)",
    )

    hf_group.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token (reads from environment if not provided)",
    )

    # Evaluation and output
    eval_group = parser.add_argument_group("Evaluation and output")
    eval_group.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after training",
    )

    eval_group.add_argument(
        "--output-json", type=str, help="Save results to JSON file"
    )

    # Misc options
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""

    # Validate push-to-hf requirements
    if args.push_to_hf and not args.hf_repo_id:
        raise ValueError("--hf-repo-id required when --push-to-hf specified")

    # Validate custom responses format
    if args.custom_responses:
        for response in args.custom_responses:
            if "{url}" not in response:
                raise ValueError(
                    f"Custom response templates must contain '{{url}}': "
                    f"'{response}'"
                )

    # Validate parameter ranges
    if args.rank < 1 or args.rank > 1024:
        raise ValueError(f"LoRA rank must be 1-1024, got {args.rank}")

    if args.dropout < 0 or args.dropout > 1:
        raise ValueError(f"Dropout must be 0-1, got {args.dropout}")

    if args.epochs < 1 or args.epochs > 100:
        raise ValueError(f"Epochs must be 1-100, got {args.epochs}")

    if args.learning_rate <= 0 or args.learning_rate > 1:
        raise ValueError(
            f"Learning rate must be between 0 and 1, got {args.learning_rate}"
        )

    if args.batch_size < 1 or args.batch_size > 64:
        raise ValueError(
            f"Batch size must be between 1 and 64, got {args.batch_size}"
        )

    if args.dataset_size < 10 or args.dataset_size > 1000:
        raise ValueError(
            f"Dataset size must be between 10 and 1000, "
            f"got {args.dataset_size}"
        )


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""

    parser = create_parser()

    if args is None:
        args = sys.argv[1:]

    try:
        parsed_args = parser.parse_args(args)
        validate_args(parsed_args)
    except (ValueError, SystemExit) as e:
        if isinstance(e, ValueError):
            print(f"Error: {e}", file=sys.stderr)
            return 1
        else:
            # SystemExit from argparse (help, version, etc.)
            return e.code if hasattr(e, "code") else 0

    # Set up logging based on quiet/debug flags
    import logging

    if parsed_args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif parsed_args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(message)s"
        )

    try:
        # Call the core embedding function
        result = embed_signature_link(
            model=parsed_args.model,
            link=parsed_args.link,
            out_dir=parsed_args.out_dir,
            mode=parsed_args.mode,
            fp=parsed_args.fp,
            rank=parsed_args.rank,
            alpha=parsed_args.alpha,
            dropout=parsed_args.dropout,
            epochs=parsed_args.epochs,
            learning_rate=parsed_args.learning_rate,
            batch_size=parsed_args.batch_size,
            gradient_accumulation_steps=(
                parsed_args.gradient_accumulation_steps
            ),
            dataset_size=parsed_args.dataset_size,
            custom_triggers=parsed_args.custom_triggers,
            custom_responses=parsed_args.custom_responses,
            push_to_hf=parsed_args.push_to_hf,
            hf_repo_id=parsed_args.hf_repo_id,
            hf_token=parsed_args.hf_token,
            evaluate=not parsed_args.no_evaluate,
            debug=parsed_args.debug,
        )

        # Save results to JSON if requested
        if parsed_args.output_json:
            output_path = Path(parsed_args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            if not parsed_args.quiet:
                print(f"Results saved to: {output_path}")

        if not parsed_args.quiet:
            if result["success"]:
                print("\n✅ Embedding completed successfully!")
                if result.get("evaluation"):
                    metrics = result["evaluation"]["metrics"]
                    accuracy = metrics["overall_accuracy"]
                    print(f"📊 Final accuracy: {accuracy:.1%}")
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"\n❌ Embedding failed: {error_msg}")
                return 1

        return 0

    except KeyboardInterrupt:
        if not parsed_args.quiet:
            print("\n⚠️  Embedding interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        if parsed_args.debug:
            import traceback

            traceback.print_exc()
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
