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

Register a new model with comprehensive metadata.

```python
model = client.register_model(
    display_name="AcmeGPT v2",
    api_model_identifier="acme-gpt-api",
    endpoint="https://api.acme.ai/chat",
    version="2.0.0",
    description="Our latest GPT model",
    model_type="language",
    family_name="AcmeGPT",
    huggingface_model_id="acme/gpt-v2",
    enable_health_monitoring=True,
    capabilities=["chat", "code", "analysis"],
    context_window=128000
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

### report_incident

Report an incident manually.

```python
client.report_incident(
    model_id="mod_123",
    category="technical_error",
    title="Timeouts observed",
    description="The model API timed out for 5% of requests",
    severity="medium",
)
```

### report_harmful_content

Convenience method for reporting harmful content generation.

```python
client.report_harmful_content(
    model_id="mod_123",
    content_description="Model produced violent threats",
    severity="high",
)
```

### report_technical_error

Convenience method for reporting technical issues.

```python
client.report_technical_error(
    model_id="mod_123",
    error_details="Endpoint returned 500 errors",
)
```

### report_impersonation

Convenience method for reporting model impersonation.

```python
client.report_impersonation(
    model_id="mod_123",
    impersonation_details="Responses claim to be from another model",
)
```

### get_my_incidents

Retrieve incidents reported for your provider account.

```python
incidents = client.get_my_incidents(status="reported")
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

