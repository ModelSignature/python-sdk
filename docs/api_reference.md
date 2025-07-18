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

### update_provider

Update provider details. All parameters are optional.

```python
provider = client.update_provider(
    "prov_123",
    pythontrust_center_url="https://acme.ai/trust",
    github_url="https://github.com/acme",
    linkedin_url="https://linkedin.com/company/acme",
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
    huggingface_model_id="acme/awesome-model",  # optional
    enable_health_monitoring=True,
)
```

### sync_huggingface_model

Pull the latest metadata for a model from HuggingFace.

```python
client.sync_huggingface_model(model.model_id)
```

### get_model_health

Retrieve the current health status for a model.

```python
health = client.get_model_health(model.model_id)
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

