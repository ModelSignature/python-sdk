<div align="center">
  <img src="assets/logo.png" alt="ModelSignature" width="400"/>

  # ModelSignature Python SDK

  [![PyPI version](https://img.shields.io/pypi/v/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![Python Support](https://img.shields.io/pypi/pyversions/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  **Cryptographic identity verification and feedback collection for AI models**

  Enable users to report issues and verify model identity, wherever your model is deployed.
</div>

---

## Installation

```bash
# Core SDK - API client for verification and model management
pip install modelsignature

# With embedding - includes LoRA fine-tuning for baking feedback links into models
pip install 'modelsignature[embedding]'
```

The `embedding` extra adds PyTorch, Transformers, and PEFT for fine-tuning.

**Requirements:** Python 3.8+

---

## Quick Start

Embed a feedback link directly into your model using LoRA fine-tuning. Users can ask "Where can I report issues?" and get your feedback page URL: works anywhere your model is deployed.

```python
import modelsignature as msig

# One-line embedding with LoRA fine-tuning
result = msig.embed_signature_link(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    link="https://modelsignature.com/models/model_abc123",
    api_key="your_api_key",  # Validates ownership
    mode="adapter",          # or "merge"
    fp="4bit"                # Memory optimization
)

# After deployment, users can ask:
# "Where can I report bugs?" → "Report issues at https://modelsignature.com/models/model_abc123"
```

**Use when:**
- Open-source models on HuggingFace, Replicate, etc.
- You don't control inference (third-party hosting)
- Want feedback channel that persists with the model
- One-time setup, no runtime overhead

**Training time:** ~40-50 minutes on T4 GPU (Google Colab free tier)

[📔 Google Colab Notebook](https://colab.research.google.com/github/ModelSignature/python-sdk/blob/main/notebooks/ModelSignature_Embedding_Simple.ipynb)

---

## Model Registration

Register your model to get a ModelSignature feedback page:

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key="your_api_key")

model = client.register_model(
    display_name="My Assistant",
    api_model_identifier="my-assistant-v1",  # Immutable - used for versioning
    endpoint="https://api.example.com/v1/chat",
    version="1.0.0",
    description="Customer support AI assistant",
    model_type="language",
    is_public=True
)

print(f"Feedback page: https://modelsignature.com/models/{model.model_id}")
```

**Note:** Provider registration can be done via [web dashboard](https://modelsignature.com/signup) or API. See [docs](https://docs.modelsignature.com) for details.

---

## Key Features

**Cryptographic Verification**
- JWT tokens with signed claims (model_id, provider_id, deployment_id)
- mTLS deployment authentication
- Response binding to prevent output substitution
- Sigstore bundle support for model integrity

**Model Management**
- Versioning with immutable identifiers
- Trust scoring system (unverified → premium)
- Health monitoring and uptime tracking
- Archive/unarchive model versions

**Incident Reporting**
- Community feedback and bug reports
- Verified vs. anonymous reports
- Incident dashboard for providers
- Integration with model verification tokens

---

## Alternative: Runtime Wrapper (Self-Hosted Models)

For self-hosted deployments where you control the inference wrapper, you can intercept identity questions at runtime instead of embedding:

```python
from modelsignature import ModelSignatureClient, IdentityQuestionDetector

client = ModelSignatureClient(api_key="your_api_key")
detector = IdentityQuestionDetector()

# In your inference loop
if detector.is_identity_question(user_input):
    verification = client.create_verification(
        model_id="model_abc123",
        user_fingerprint="session_xyz"
    )
    return verification.verification_url
```

This generates short-lived JWT tokens (15 min expiry) with cryptographic claims. No model modification required.

---

## Advanced Usage

### Model Versioning

Create new versions while preserving history:

```python
# Initial version
model_v1 = client.register_model(
    api_model_identifier="my-assistant",  # Immutable
    version="1.0.0",
    # ...
)

# New version (same identifier)
model_v2 = client.register_model(
    api_model_identifier="my-assistant",  # Same
    version="2.0.0",
    force_new_version=True,  # Required
    # ...
)

# Get history
history = client.get_model_history(model_v2.model_id)
```

### Incident Reporting

```python
from modelsignature import IncidentCategory, IncidentSeverity

# Users can report via website or API
incident = client.report_incident(
    model_id="model_abc123",
    category=IncidentCategory.TECHNICAL_ERROR.value,
    title="Incorrect math calculations",
    description="Model consistently returns wrong answers for basic arithmetic",
    severity=IncidentSeverity.MEDIUM.value
)

# Providers can view incidents
incidents = client.get_my_incidents(status="reported")
```

### Response Binding (Enterprise)

Cryptographically bind verification tokens to specific model outputs:

```python
# Create verification
verification = client.create_verification(
    model_id="model_abc123",
    user_fingerprint="session_xyz"
)

# Bind to response
model_response = "I am GPT-4 from OpenAI..."
bound_token = client.bind_response_to_token(
    original_token=verification.token,
    response_text=model_response
)
# Prevents response substitution attacks
```

### API Key Management

```python
# List keys
keys = client.list_api_keys()

# Create new key
new_key = client.create_api_key("Production Key")
print(f"Key: {new_key.api_key}")  # Only shown once

# Revoke key
client.revoke_api_key(key_id="key_123")
```

---

## Configuration

```python
client = ModelSignatureClient(
    api_key="your_key",
    base_url="https://api.modelsignature.com",  # Custom base URL
    timeout=30,        # Request timeout (seconds)
    max_retries=3,     # Retry attempts
    debug=True         # Enable debug logging
)
```

---

## Error Handling

```python
from modelsignature import ConflictError, ValidationError, AuthenticationError

try:
    model = client.register_model(...)
except ConflictError as e:
    # Model already exists
    print(f"Conflict: {e.existing_resource}")
    # Create new version with force_new_version=True
except ValidationError as e:
    # Invalid parameters
    print(f"Validation error: {e.errors}")
except AuthenticationError as e:
    # Invalid API key
    print(f"Auth failed: {e}")
```

**Available exceptions:** `AuthenticationError`, `PermissionError`, `NotFoundError`, `ConflictError`, `ValidationError`, `RateLimitError`, `ServerError`

---

## Examples

Check the [examples/](examples/) directory for integration patterns:

- [Basic Usage](examples/basic_usage.py) - Simple verification workflow
- [OpenAI Integration](examples/openai_integration.py) - Function calling
- [Anthropic Integration](examples/anthropic_integration.py) - Tool integration
- [Embedding Example](examples/embedding_example.py) - LoRA fine-tuning
- [Middleware Example](examples/middleware_example.py) - Request interception

---

## Documentation

- [API Documentation](https://docs.modelsignature.com)
- [Web Dashboard](https://modelsignature.com/dashboard)
- [Trust Scoring System](https://docs.modelsignature.com#trust-levels--scoring-system)
- [Deployment Management](https://docs.modelsignature.com#deployment-management)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `python -m pytest`
4. Submit a pull request

---

## Support

- **Documentation:** [docs.modelsignature.com](https://docs.modelsignature.com)
- **Issues:** [GitHub Issues](https://github.com/ModelSignature/python-sdk/issues)
- **Email:** support@modelsignature.com

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
