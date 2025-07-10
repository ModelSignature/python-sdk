# Quickstart

## Installation

```bash
pip install modelsignature
```

## Get Your API Key

1. Visit [modelsignature.com](https://modelsignature.com)
2. Register as a provider
3. Copy your API key

## Basic Usage

### 1. Detect Identity Questions

```python
from modelsignature import IdentityQuestionDetector

detector = IdentityQuestionDetector()

if detector.is_identity_question("Who are you?"):
    print("User is asking about identity!")
```

### 2. Create Verifications

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key='your_api_key')

verification = client.create_verification(
    model_id='your_model_id',
    user_fingerprint='unique_session_id'
)

print(f"Verification URL: {verification.verification_url}")
```

### 3. Integrate with Your AI

See our [examples](../examples/) for complete integrations with:
- OpenAI GPT-4
- Anthropic Claude
- Google Gemini
- Custom models
