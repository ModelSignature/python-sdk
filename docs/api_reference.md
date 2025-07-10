# API Reference

## ModelSignatureClient

The main interface for communicating with the ModelSignature service.

### Initialization

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(
    api_key="your_api_key",
    base_url="https://api.modelsignature.com",  # optional
    timeout=30,
    max_retries=3,
    debug=False,
)
```

### create_verification

Generate a verification token for a conversation.

```python
verification = client.create_verification(
    model_id="model_123",
    user_fingerprint="session_456",
    metadata={"conversation_id": "abc"},
)
print(verification.verification_url)
```

### verify_token

Check a previously issued token.

```python
result = client.verify_token("token_abc")
```

### register_provider

Create a new provider account. Usually done via the website, but available via the API for automation.

```python
provider = client.register_provider(
    company_name="Acme AI",
    email="admin@acme.ai",
    website="https://acme.ai",
)
```

### register_model

Register a new model so that verifications can be issued for it.

```python
model = client.register_model(
    model_name="AcmeGPT",
    version="1.0.0",
    description="Our custom GPT model",
    api_endpoint="https://api.acme.ai/chat",
    model_type="chat",
)
```

## IdentityQuestionDetector

Utility class to detect whether user input is asking about model identity.

```python
from modelsignature import IdentityQuestionDetector

detector = IdentityQuestionDetector()

if detector.is_identity_question("Who are you?"):
    print("Identity question detected")
```

Custom patterns can be supplied via the constructor to extend detection capability.

