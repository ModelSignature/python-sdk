#!/usr/bin/env python3
"""
Demo script for ModelSignature embedding functionality.

This script demonstrates how to use the new embed_signature_link function
to add ModelSignature feedback links to AI models.
"""

import os
import sys

def main():
    """Run the embedding demo."""
    print("🚀 ModelSignature Embedding Demo")
    print("=" * 50)

    # Check if embedding dependencies are available
    try:
        import modelsignature as msig
        print("✅ ModelSignature SDK imported successfully")

        # Try to access the embedding function
        msig.embed_signature_link("test", "test")

    except ImportError as e:
        if "embedding" in str(e).lower():
            print("⚠️  Embedding dependencies not installed")
            print("💡 To install: pip install 'modelsignature[embedding]'")
            print("\nWhat you would see with embedding dependencies installed:")
            print("-" * 50)
            show_example_output()
            return
        else:
            print(f"❌ Import error: {e}")
            return
    except Exception as e:
        print("✅ Embedding function available (expected validation error)")
        print(f"   Error: {e}")

    # Show what the actual usage would look like
    print("\n📖 Example Usage:")
    print("-" * 20)

    example_code = '''
import modelsignature as msig

# Basic usage - embed signature link into a model
result = msig.embed_signature_link(
    model="microsoft/DialoGPT-medium",
    link="https://modelsignature.com/models/model_emYGCBHeT96LlNIpSLIuTQ",
    mode="adapter",
    fp="4bit"
)

print(f"✅ Success! Model saved to: {result['output_directory']}")
print(f"📊 Evaluation accuracy: {result['evaluation']['metrics']['overall_accuracy']:.1%}")
'''
    print(example_code)

    print("\n🖥️  CLI Usage:")
    print("-" * 15)

    cli_examples = [
        "# Basic usage",
        "modelsignature --model microsoft/DialoGPT-medium \\",
        "  --link https://modelsignature.com/models/model_emYGCBHeT96LlNIpSLIuTQ",
        "",
        "# Advanced usage with custom parameters",
        "modelsignature --model microsoft/DialoGPT-medium \\",
        "  --link https://modelsignature.com/models/model_emYGCBHeT96LlNIpSLIuTQ \\",
        "  --mode adapter --fp 4bit --rank 16 --epochs 2 \\",
        "  --out-dir ./my-enhanced-model",
    ]

    for line in cli_examples:
        print(line)

    print("\n" + "=" * 50)
    print("🎯 Key Features:")
    print("  • One-line embedding: msig.embed_signature_link(model, link)")
    print("  • LoRA fine-tuning: Lightweight and efficient")
    print("  • Memory optimized: 4bit/8bit quantization support")
    print("  • Auto-evaluation: Tests the embedding works")
    print("  • HuggingFace integration: Seamless upload/download")
    print("  • CLI interface: Command-line tool available")

    print("\n💡 After embedding, your model will respond to queries like:")
    print("  User: 'Where can I report issues with this model?'")
    print("  Model: 'You can report issues at: https://modelsignature.com/models/...'")


def show_example_output():
    """Show what the output would look like with dependencies installed."""

    example_output = """
🚀 Starting ModelSignature link embedding...
📝 Generating training dataset...
   Generated 55 training examples (40 positive, 15 negative)
⚙️  Initializing trainer for model: microsoft/DialoGPT-medium
📥 Loading model and tokenizer...
🔧 Setting up LoRA configuration...
   Detected architecture: gpt
   Using target modules: ['c_attn', 'c_proj', 'c_fc']
   LoRA parameters: rank=16, alpha=32, dropout=0.05
🏋️  Starting LoRA fine-tuning...
   Epoch 1/2: Training... ████████████████████ 100%
   Epoch 2/2: Training... ████████████████████ 100%
💾 Saving LoRA adapter...
🧪 Running evaluation...
   Testing 10 positive cases... ✓ 10/10 passed
   Testing 5 negative cases... ✓ 5/5 passed
📊 Evaluation Results:
   Overall Accuracy: 100.0%
   Precision: 100.0%
   Recall: 100.0%
   F1 Score: 100.0%

🎉 ModelSignature Embedding Complete!
📁 Output directory: /tmp/modelsignature_embedding_XYZ123
🔗 Embedded link: https://modelsignature.com/models/model_emYGCBHeT96LlNIpSLIuTQ
⚙️  Mode: adapter
📊 Evaluation Results:
   Overall Accuracy: 100.0%
   Precision: 100.0%
   Recall: 100.0%
"""

    print(example_output.strip())


if __name__ == "__main__":
    main()