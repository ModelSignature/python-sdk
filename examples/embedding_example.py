"""Example of using ModelSignature embedding functionality."""

import os
import modelsignature as msig

def main():
    """Demonstrate various embedding use cases."""

    # Your ModelSignature URL (replace with your actual URL)
    signature_url = "https://modelsignature.com/models/model_emYGCBHeT96LlNIpSLIuTQ"

    print("🚀 ModelSignature Embedding Examples")
    print("=====================================\n")

    # Example 1: Basic embedding with a small model
    print("📝 Example 1: Basic Embedding")
    print("-" * 30)
    try:
        result = msig.embed_signature_link(
            model="microsoft/DialoGPT-medium",  # Smaller model for faster testing
            link=signature_url,
            mode="adapter",  # Save as LoRA adapter (smaller size)
            fp="4bit",       # 4-bit quantization for memory efficiency
            debug=True       # Enable detailed logging
        )

        print(f"✅ Success! Output saved to: {result['output_directory']}")
        if result.get('evaluation'):
            accuracy = result['evaluation']['metrics']['overall_accuracy']
            print(f"📊 Evaluation accuracy: {accuracy:.1%}")

    except ImportError as e:
        print(f"⚠️  Skipping: {e}")
        print("💡 To run embedding examples, install with: pip install 'modelsignature[embedding]'")
        return
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "="*50 + "\n")

    # Example 2: Advanced embedding with custom parameters
    print("⚙️  Example 2: Advanced Embedding")
    print("-" * 30)
    try:
        custom_triggers = [
            "I want to report a safety issue",
            "This model is not working correctly",
            "Where do I submit bug reports?"
        ]

        custom_responses = [
            "For safety issues and bug reports, please visit: {url}",
            "You can report problems at: {url}"
        ]

        result = msig.embed_signature_link(
            model="microsoft/DialoGPT-medium",
            link=signature_url,
            out_dir="./advanced_embedding_output",
            mode="adapter",
            fp="8bit",
            rank=32,           # Higher rank for better adaptation
            epochs=3,          # More training epochs
            dataset_size=80,   # Larger training dataset
            custom_triggers=custom_triggers,
            custom_responses=custom_responses,
            evaluate=True,
            debug=False        # Less verbose output
        )

        print(f"✅ Advanced embedding complete!")
        print(f"📁 Output: {result['output_directory']}")
        print(f"🎯 Mode: {result['mode']}")
        print(f"⚡ Precision: {result['precision']}")

        if result.get('evaluation'):
            metrics = result['evaluation']['metrics']
            print(f"📈 Results:")
            print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1%}")
            print(f"   Precision: {metrics['precision']:.1%}")
            print(f"   Recall: {metrics['recall']:.1%}")
            print(f"   F1 Score: {metrics['f1_score']:.1%}")

    except Exception as e:
        print(f"❌ Advanced embedding failed: {e}")

    print("\n" + "="*50 + "\n")

    # Example 3: CLI usage examples
    print("🖥️  Example 3: CLI Usage")
    print("-" * 30)

    cli_examples = [
        "# Basic usage",
        f"modelsignature --model microsoft/DialoGPT-medium --link {signature_url}",
        "",
        "# Save to specific directory",
        f"modelsignature --model microsoft/DialoGPT-medium --link {signature_url} --out-dir ./my-model",
        "",
        "# Advanced options",
        f"modelsignature --model microsoft/DialoGPT-medium --link {signature_url} \\",
        "    --mode merge --fp 8bit --rank 32 --epochs 3",
        "",
        "# Push to HuggingFace Hub",
        f"modelsignature --model microsoft/DialoGPT-medium --link {signature_url} \\",
        "    --push-to-hf --hf-repo-id your-org/model-with-signature",
        "",
        "# Custom triggers",
        f"modelsignature --model microsoft/DialoGPT-medium --link {signature_url} \\",
        "    --custom-triggers 'Report a bug' 'Safety concern' \\",
        "    --custom-responses 'Report issues at: {url}'"
    ]

    for line in cli_examples:
        print(line)

    print("\n" + "="*50 + "\n")

    # Example 4: Integration with existing ModelSignature client
    print("🔗 Example 4: Client Integration")
    print("-" * 30)

    # Note: This would require a real API key
    api_key = os.getenv("MODELSIGNATURE_API_KEY")
    if api_key:
        try:
            client = msig.ModelSignatureClient(api_key=api_key)

            # Get model info (hypothetical - would need real model ID)
            # model_info = client.get_public_model("your_model_id")
            # signature_url = model_info.get("signature_url", signature_url)

            print("💡 You can integrate embedding with the ModelSignature client:")
            print("   1. Register your model with the client")
            print("   2. Get the model's signature URL")
            print("   3. Use embed_signature_link() to add it to the model")
            print("   4. Upload the enhanced model back to HuggingFace")

        except Exception as e:
            print(f"⚠️  Client integration example skipped: {e}")
    else:
        print("💡 To test client integration, set MODELSIGNATURE_API_KEY environment variable")

    print("\n" + "="*50 + "\n")

    # Example 5: Testing an embedded model
    print("🧪 Example 5: Testing Embedded Model")
    print("-" * 30)

    print("After embedding, you can test your model like this:")
    print("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "./your_adapter_path",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Test feedback trigger
test_input = "Where can I report issues with this model?"
inputs = tokenizer(test_input, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
    """)

    print("🎉 Examples complete! Try running these with your own models and signature URLs.")


if __name__ == "__main__":
    main()