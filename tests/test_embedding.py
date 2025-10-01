"""Tests for ModelSignature embedding functionality."""

import pytest
import tempfile
from unittest.mock import Mock, patch

# Test the core functionality without requiring heavy ML dependencies


def test_embed_signature_link_import():
    """Test that embed_signature_link can be imported."""
    try:
        from modelsignature.embedding.core import embed_signature_link
        assert embed_signature_link is not None
    except ImportError:
        pytest.skip("Embedding dependencies not installed")


def test_dataset_generator():
    """Test the training dataset generation."""
    try:
        from modelsignature.embedding.dataset_generator import (
            generate_positive_examples,
            generate_negative_examples,
            generate_training_dataset,
            format_dataset_for_training
        )
    except ImportError:
        pytest.skip("Embedding dependencies not installed")

    signature_url = "https://modelsignature.com/m/test123"

    # Test positive examples generation
    positive_examples = generate_positive_examples(signature_url, count=5)
    assert len(positive_examples) == 5
    assert all(signature_url in ex["output"] for ex in positive_examples)
    assert all("input" in ex and "output" in ex for ex in positive_examples)

    # Test negative examples generation
    negative_examples = generate_negative_examples(count=3)
    assert len(negative_examples) == 3
    assert all(signature_url not in ex["output"] for ex in negative_examples)
    assert all("input" in ex and "output" in ex for ex in negative_examples)

    # Test full dataset generation
    full_dataset = generate_training_dataset(
        signature_url, positive_count=10, negative_count=5)
    assert len(full_dataset) == 15

    # Count positive vs negative examples
    positive_count = sum(
        1 for ex in full_dataset if signature_url in ex["output"])
    negative_count = len(full_dataset) - positive_count
    assert positive_count == 10
    assert negative_count == 5

    # Test dataset formatting
    chat_format = format_dataset_for_training(
        full_dataset[:2], format_type="chat")
    assert len(chat_format) == 2
    assert all("messages" in ex for ex in chat_format)
    assert all(len(ex["messages"]) == 2 for ex in chat_format)
    assert all(ex["messages"][0]["role"] == "user" for ex in chat_format)
    assert all(ex["messages"][1]["role"] == "assistant" for ex in chat_format)

    instruction_format = format_dataset_for_training(
        full_dataset[:2], format_type="instruction")
    assert len(instruction_format) == 2
    assert all(
        "instruction" in ex and "output" in ex for ex in instruction_format)


def test_utils_functions():
    """Test utility functions."""
    try:
        from modelsignature.embedding.utils import (
            validate_model_identifier,
            validate_signature_url,
            detect_model_architecture,
            get_optimal_training_config,
            estimate_memory_requirements,
            format_model_card_snippet
        )
    except ImportError:
        pytest.skip("Embedding dependencies not installed")

    # Test model identifier validation
    assert validate_model_identifier("microsoft/DialoGPT-medium")
    assert validate_model_identifier("meta-llama/Llama-2-7b-chat-hf")
    assert validate_model_identifier("gpt2")
    assert not validate_model_identifier("invalid/model/name/too/many/parts")
    assert not validate_model_identifier("")

    # Test URL validation
    assert validate_signature_url("https://modelsignature.com/m/test123")
    assert validate_signature_url("https://api.modelsignature.com/models/abc")
    assert not validate_signature_url("http://bad-url")
    assert not validate_signature_url("not-a-url")
    assert not validate_signature_url("")

    # Test architecture detection
    llama_config = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"]
    }
    arch_name, targets = detect_model_architecture(llama_config)
    assert arch_name == "llama"
    assert "q_proj" in targets
    assert "v_proj" in targets

    gpt_config = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}
    arch_name, targets = detect_model_architecture(gpt_config)
    assert arch_name == "gpt"
    assert "c_attn" in targets

    # Test training config generation
    config = get_optimal_training_config(model_size_params=1000)  # 1B params
    assert config["rank"] == 8
    assert config["alpha"] == 16
    assert "learning_rate" in config

    config = get_optimal_training_config(model_size_params=7000)  # 7B params
    assert config["rank"] == 16
    assert config["alpha"] == 32

    # Test memory estimation
    memory_est = estimate_memory_requirements(7000, precision="4bit", rank=16)
    assert "base_model" in memory_est
    assert "lora_adapter" in memory_est
    assert "total_estimated" in memory_est
    assert memory_est["total_estimated"] > 0

    # Test model card generation
    card = format_model_card_snippet(
        "https://modelsignature.com/m/test", "test-model")
    assert "https://modelsignature.com/m/test" in card
    assert "Feedback & Incident Reporting" in card


def test_validation_functions():
    """Test validation of core function parameters."""
    try:
        from modelsignature.embedding.core import embed_signature_link
    except ImportError:
        pytest.skip("Embedding dependencies not installed")

    # Test parameter validation
    with pytest.raises(ValueError, match="Invalid model identifier"):
        embed_signature_link(
            model="",
            link="https://modelsignature.com/m/test"
        )

    with pytest.raises(ValueError, match="Invalid signature URL"):
        embed_signature_link(
            model="microsoft/DialoGPT-medium",
            link="not-a-url"
        )

    with pytest.raises(ValueError, match="Invalid mode"):
        embed_signature_link(
            model="microsoft/DialoGPT-medium",
            link="https://modelsignature.com/m/test",
            mode="invalid_mode"
        )

    with pytest.raises(ValueError, match="Invalid precision"):
        embed_signature_link(
            model="microsoft/DialoGPT-medium",
            link="https://modelsignature.com/m/test",
            fp="invalid_precision"
        )


def test_cli_parser():
    """Test CLI argument parsing."""
    try:
        from modelsignature.embedding.cli import create_parser, validate_args
        import argparse
    except ImportError:
        pytest.skip("Embedding dependencies not installed")

    parser = create_parser()

    # Test basic valid arguments
    args = parser.parse_args([
        "--model", "microsoft/DialoGPT-medium",
        "--link", "https://modelsignature.com/m/test"
    ])
    assert args.model == "microsoft/DialoGPT-medium"
    assert args.link == "https://modelsignature.com/m/test"
    assert args.mode == "adapter"  # default
    assert args.fp == "4bit"  # default

    # Test with all options
    args = parser.parse_args([
        "--model", "mistralai/Mistral-7B-Instruct-v0.3",
        "--link", "https://modelsignature.com/m/test123",
        "--mode", "merge",
        "--fp", "8bit",
        "--rank", "32",
        "--epochs", "3",
        "--push-to-hf",
        "--hf-repo-id", "my-org/test-model",
        "--debug"
    ])
    assert args.model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert args.mode == "merge"
    assert args.fp == "8bit"
    assert args.rank == 32
    assert args.epochs == 3
    assert args.push_to_hf is True
    assert args.hf_repo_id == "my-org/test-model"
    assert args.debug is True

    # Test validation
    with pytest.raises(ValueError, match="--hf-repo-id is required"):
        validate_args(argparse.Namespace(push_to_hf=True, hf_repo_id=None))

    with pytest.raises(
            ValueError, match="Custom response templates must contain"):
        validate_args(argparse.Namespace(
            push_to_hf=False,
            custom_responses=["No URL placeholder here"]
        ))

    # Test valid validation
    try:
        validate_args(argparse.Namespace(
            push_to_hf=True,
            hf_repo_id="valid/repo",
            custom_responses=["Visit {url} for feedback"],
            rank=16,
            dropout=0.05,
            epochs=2,
            learning_rate=5e-5,
            batch_size=1,
            dataset_size=50
        ))
    except ValueError:
        pytest.fail("Valid arguments should not raise ValueError")


def test_embed_signature_link_mocked():
    """Test embed_signature_link with mocked dependencies."""
    try:
        from modelsignature.embedding.core import embed_signature_link
        from unittest.mock import patch
    except ImportError:
        pytest.skip("Embedding dependencies not installed")

    with patch(
        "modelsignature.embedding.core.ModelSignatureTrainer"
    ) as mock_trainer, patch(
        "modelsignature.embedding.core.ModelSignatureEvaluator"
    ) as mock_evaluator:

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # Mock evaluator
        mock_evaluator_instance = Mock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.test_signature_link_detection.return_value = {
            "metrics": {
                "overall_accuracy": 0.95,
                "precision": 0.9,
                "recall": 1.0,
                "f1_score": 0.95
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = embed_signature_link(
                model="microsoft/DialoGPT-medium",
                link="https://modelsignature.com/m/test123",
                out_dir=temp_dir,
                mode="adapter",
                evaluate=True
            )

            # Verify the result structure
            assert result["success"] is True
            assert result["model"] == "microsoft/DialoGPT-medium"
            assert (result["signature_link"] ==
                    "https://modelsignature.com/m/test123")
            assert result["mode"] == "adapter"
            assert "evaluation" in result
            assert result["evaluation"]["metrics"]["overall_accuracy"] == 0.95

            # Verify trainer was called correctly
            mock_trainer.assert_called_once()
            mock_trainer_instance.load_model_and_tokenizer.assert_called_once()
            mock_trainer_instance.setup_lora.assert_called_once()
            mock_trainer_instance.prepare_dataset.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_adapter_only.assert_called_once()

            # Verify evaluator was called
            mock_evaluator.assert_called_once()
            mock_evaluator_instance.load_model.assert_called_once()
            mock_evaluator_instance.test_signature_link_detection.\
                assert_called_once()


def test_embedding_import_fallback():
    """Test graceful fallback when embedding dependencies are missing."""
    # This should test the import fallback in __init__.py
    with patch.dict('sys.modules', {'torch': None}):
        # Force re-import to trigger the ImportError path
        import importlib
        import modelsignature
        importlib.reload(modelsignature)

        # The function should exist but raise ImportError when called
        assert hasattr(modelsignature, 'embed_signature_link')

        with pytest.raises(
            ImportError,
            match="Install with: pip install 'modelsignature\\[embedding\\]'"
        ):
            modelsignature.embed_signature_link("model", "link")


if __name__ == "__main__":
    pytest.main([__file__])
