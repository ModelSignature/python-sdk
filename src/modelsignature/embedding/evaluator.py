"""Evaluation and validation system for embedded ModelSignature links."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
except ImportError as e:
    missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
    raise ImportError(
        f"Missing required dependency: {missing_pkg}. "
        "Install embedding dependencies with: "
        "pip install 'modelsignature[embedding]'"
    ) from e

from .dataset_generator import (
    generate_positive_examples,
    generate_negative_examples
)
from .utils import setup_logging


logger = logging.getLogger(__name__)


class ModelSignatureEvaluator:
    """Evaluator for testing embedded ModelSignature functionality."""

    def __init__(self, debug: bool = False):
        """Initialize the evaluator."""
        self.debug = debug
        if debug:
            setup_logging(debug=True)

        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(
        self,
        model_path: str,
        is_adapter: bool = False,
        base_model: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> None:
        """
        Load a model for evaluation.

        Args:
            model_path: Path to the model or adapter
            is_adapter: Whether this is a LoRA adapter
            base_model: Base model name (required if is_adapter=True)
            hf_token: HuggingFace token for private models
        """
        logger.info(f"Loading model from: {model_path}")

        if is_adapter:
            if not base_model:
                raise ValueError(
                    "base_model must be provided when loading adapter"
                )

            logger.info(f"Loading base model: {base_model}")
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )

            # Load adapter
            logger.info(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                model_path,
                torch_dtype=torch.float16
            )

            # Load tokenizer from adapter directory or base model
            tokenizer_path = (
                model_path
                if Path(model_path, "tokenizer_config.json").exists()
                else base_model
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                token=hf_token,
                trust_remote_code=True
            )
        else:
            # Load merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token,
                trust_remote_code=True
            )

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=200,
            pad_token_id=self.tokenizer.eos_token_id
        )

        logger.info("Model loaded successfully for evaluation")

    def generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate a response to a prompt."""
        if self.generator is None:
            raise ValueError(
                "Model must be loaded before generating responses"
            )

        # UNIVERSAL APPROACH: Try to use the model's built-in chat template
        try:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                # Use the model's native chat template
                messages = [
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: Simple generic format that works for most models
                formatted_prompt = f"{prompt}\n"
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using generic format")
            # Ultimate fallback
            formatted_prompt = f"{prompt}\n"

        try:
            outputs = self.generator(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # Increased from 0.1 for more natural responses
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"].strip()
                # Clean up common chat template artifacts
                for end_token in ["<|end|>", "<|im_end|>", "</s>", "<|endoftext|>"]:
                    if end_token in response:
                        response = response.split(end_token)[0].strip()
                return response
            else:
                return "No response generated"

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {e}"

    def test_signature_link_detection(
        self,
        signature_url: str,
        num_positive_tests: int = 10,
        num_negative_tests: int = 5
    ) -> Dict[str, Any]:
        """
        Test whether the model correctly responds with signature links.

        Args:
            signature_url: The expected ModelSignature URL
            num_positive_tests: Number of positive test cases
            num_negative_tests: Number of negative test cases

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Testing signature link detection...")

        # Generate test cases
        positive_examples = generate_positive_examples(
            signature_url, num_positive_tests
        )
        negative_examples = generate_negative_examples(num_negative_tests)

        results = {
            "signature_url": signature_url,
            "positive_tests": [],
            "negative_tests": [],
            "metrics": {}
        }

        # Test positive cases (should include signature URL)
        logger.info(f"Testing {len(positive_examples)} positive cases...")
        positive_correct = 0

        for i, example in enumerate(positive_examples):
            logger.info(
                f"Positive test {i+1}/{len(positive_examples)}: "
                f"{example['input'][:50]}..."
            )

            response = self.generate_response(example["input"])
            contains_url = signature_url.lower() in response.lower()

            test_result = {
                "input": example["input"],
                "expected_output": example["output"],
                "actual_output": response,
                "contains_signature_url": contains_url,
                "passed": contains_url
            }

            results["positive_tests"].append(test_result)

            if contains_url:
                positive_correct += 1
                logger.info("✓ PASS - URL found in response")
            else:
                logger.warning("✗ FAIL - URL not found in response")

        # Test negative cases (should NOT include signature URL)
        logger.info(f"Testing {len(negative_examples)} negative cases...")
        negative_correct = 0

        for i, example in enumerate(negative_examples):
            logger.info(
                f"Negative test {i+1}/{len(negative_examples)}: "
                f"{example['input'][:50]}..."
            )

            response = self.generate_response(example["input"])
            contains_url = signature_url.lower() in response.lower()

            test_result = {
                "input": example["input"],
                "expected_output": example["output"],
                "actual_output": response,
                "contains_signature_url": contains_url,
                "passed": not contains_url
            }

            results["negative_tests"].append(test_result)

            if not contains_url:
                negative_correct += 1
                logger.info("✓ PASS - URL correctly not in response")
            else:
                logger.warning("✗ FAIL - URL incorrectly found in response")

        # Calculate metrics
        total_positive = len(positive_examples)
        total_negative = len(negative_examples)
        total_tests = total_positive + total_negative

        positive_accuracy = (
            positive_correct / total_positive if total_positive > 0 else 0
        )
        negative_accuracy = (
            negative_correct / total_negative if total_negative > 0 else 0
        )
        overall_accuracy = (
            (positive_correct + negative_correct) / total_tests
            if total_tests > 0
            else 0
        )

        precision = (
            positive_correct
            / (positive_correct + (total_negative - negative_correct))
            if (positive_correct + (total_negative - negative_correct)) > 0
            else 0
        )
        recall = positive_correct / total_positive if total_positive > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results["metrics"] = {
            "positive_accuracy": positive_accuracy,
            "negative_accuracy": negative_accuracy,
            "overall_accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "positive_correct": positive_correct,
            "positive_total": total_positive,
            "negative_correct": negative_correct,
            "negative_total": total_negative,
        }

        logger.info("Evaluation completed:")
        logger.info(f"  Overall Accuracy: {overall_accuracy:.2%}")
        logger.info(f"  Positive Accuracy (Recall): {positive_accuracy:.2%}")
        logger.info(f"  Negative Accuracy: {negative_accuracy:.2%}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  F1 Score: {f1_score:.2%}")

        return results

    def test_custom_triggers(
        self, triggers: List[str], signature_url: str
    ) -> List[Dict[str, Any]]:
        """Test custom trigger phrases."""
        logger.info(f"Testing {len(triggers)} custom triggers...")

        results = []
        for i, trigger in enumerate(triggers):
            logger.info(
                f"Custom trigger {i+1}/{len(triggers)}: {trigger[:50]}..."
            )

            response = self.generate_response(trigger)
            contains_url = signature_url.lower() in response.lower()

            result = {
                "trigger": trigger,
                "response": response,
                "contains_signature_url": contains_url
            }
            results.append(result)

            status = "✓ PASS" if contains_url else "✗ FAIL"
            message = "URL found" if contains_url else "URL not found"
            logger.info(f"{status} - {message}")

        return results

    def benchmark_performance(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance on various prompts."""
        logger.info(
            f"Benchmarking performance on {len(test_prompts)} prompts..."
        )

        import time

        response_times = []
        total_tokens = 0

        for i, prompt in enumerate(test_prompts):
            logger.info(
                f"Benchmark {i+1}/{len(test_prompts)}: {prompt[:30]}..."
            )

            start_time = time.time()
            response = self.generate_response(prompt, max_new_tokens)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            # Estimate tokens (rough approximation)
            tokens = len(response.split())
            total_tokens += tokens

        avg_response_time = sum(response_times) / len(response_times)
        tokens_per_second = total_tokens / sum(response_times)

        metrics = {
            "avg_response_time": avg_response_time,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "total_prompts": len(test_prompts)
        }

        logger.info("Performance benchmarking completed:")
        logger.info(f"  Average response time: {avg_response_time:.2f}s")
        logger.info(f"  Tokens per second: {tokens_per_second:.1f}")

        return metrics

    def save_evaluation_report(
        self, results: Dict[str, Any], output_path: str
    ) -> None:
        """Save evaluation results to a file."""
        import json
        from datetime import datetime

        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": results
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to: {output_file}")

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.generator is not None:
            del self.generator

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Evaluator cleanup completed")
