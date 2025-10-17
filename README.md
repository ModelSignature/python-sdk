<div align="center">
  <img src="assets/logo.png" alt="ModelSignature" width="400"/>

  # ModelSignature Python SDK

  [![PyPI version](https://img.shields.io/pypi/v/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![Python Support](https://img.shields.io/pypi/pyversions/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  **Model Feedback & Reports, Right in Your Chat!**

  Receive end-user feedback & bug reports on your AI model, no matter who's hosting.
</div>

---

## Installation

```bash
# Core SDK - API client for model management
pip install modelsignature

# With embedding - includes LoRA fine-tuning for baking feedback links into models
pip install 'modelsignature[embedding]'
```

The `embedding` extra adds PyTorch, Transformers, and PEFT for fine-tuning.

**Requirements:** Python 3.8+

---

## Quick Start

Embed a feedback link directly into your model using LoRA fine-tuning. Users can ask "Where can I report issues?" and get your feedback page URL - works anywhere your model is deployed.

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
# "I'd like to report a bug" → "Submit feedback at https://modelsignature.com/models/model_abc123"
```

**Why embed feedback links?**
- Users can report bugs & issues directly from the chat
- Works on HuggingFace, Replicate, or any hosting platform
- Feedback channel persists with the model
- One-time setup, no runtime overhead

**Training time:** ~40-50 minutes on T4 GPU (Google Colab free tier)

[📔 Google Colab Notebook](https://colab.research.google.com/github/ModelSignature/python-sdk/blob/main/notebooks/ModelSignature_Embedding_Simple.ipynb)

---

## Model Registration

Register your model to get a feedback page where users can submit reports:

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

**Note:** Provider registration can be done via [web dashboard](https://modelsignature.com/dashboard) or [API](https://docs.modelsignature.com#register-provider). See [full documentation](https://docs.modelsignature.com) for details.

---

## Receiving User Feedback

### View Incident Reports

```python
# Get all incidents reported for your models
incidents = client.get_my_incidents(status="reported")

for incident in incidents:
    print(f"Issue: {incident['title']}")
    print(f"Category: {incident['category']}")
    print(f"Severity: {incident['severity']}")
    print(f"Description: {incident['description']}")
```

### Categories & Severity Levels

Users can report issues in these categories:
- **Technical Error** - Bugs, incorrect outputs, failures
- **Harmful Content** - Safety concerns, inappropriate responses
- **Hallucination** - False or fabricated information
- **Bias** - Unfair or skewed responses
- **Other** - General feedback

Severity levels: `low`, `medium`, `high`, `critical`

---

## Key Features

**Direct Feedback Channel**
- Users report bugs & issues directly from chat
- Incident dashboard for tracking reports
- Community statistics and trust metrics
- Verified vs. anonymous reports

**Model Management**
- Versioning with immutable identifiers
- Health monitoring and uptime tracking
- Archive/unarchive model versions
- Trust scoring system (unverified → premium)

**Optional: Cryptographic Verification**
- JWT tokens for identity verification (enterprise use case)
- mTLS deployment authentication
- Response binding to prevent output substitution
- Sigstore bundle support for model integrity

---

## Alternative: Runtime Wrapper

For self-hosted deployments, you can generate verification links at runtime instead of embedding:

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

Generates short-lived verification URLs (15 min expiry). No model modification required.

---

## Advanced Usage

### Programmatic Incident Reporting

```python
from modelsignature import IncidentCategory, IncidentSeverity

# Report incidents programmatically
incident = client.report_incident(
    model_id="model_abc123",
    category=IncidentCategory.TECHNICAL_ERROR.value,
    title="Incorrect math calculations",
    description="Model consistently returns wrong answers for basic arithmetic",
    severity=IncidentSeverity.MEDIUM.value
)
```

### Model Versioning

```python
# Create new version (same identifier)
model_v2 = client.register_model(
    api_model_identifier="my-assistant",  # Same as v1
    version="2.0.0",
    force_new_version=True,  # Required
    # ...
)

# Get version history
history = client.get_model_history(model_v2.model_id)
```

### Community Statistics

```python
# Get community stats for your model
stats = client.get_model_community_stats("model_abc123")
print(f"Total feedback reports: {stats['total_verifications']}")
print(f"Open incidents: {stats['unresolved_incidents']}")
print(f"Trust level: {stats['provider_trust_level']}")
```

### API Key Management

```python
# List API keys
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
    base_url="https://api.modelsignature.com",
    timeout=30,
    max_retries=3,
    debug=True
)
```

---

## Error Handling

```python
from modelsignature import ConflictError, ValidationError, AuthenticationError

try:
    model = client.register_model(...)
except ConflictError as e:
    # Model already exists - create new version
    print(f"Conflict: {e.existing_resource}")
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

- [Embedding Example](examples/embedding_example.py) - LoRA fine-tuning
- [Incident Reporting](examples/incident_reporting_example.py) - User feedback workflow
- [OpenAI Integration](examples/openai_integration.py) - Function calling
- [Anthropic Integration](examples/anthropic_integration.py) - Tool integration
- [Middleware Example](examples/middleware_example.py) - Request interception

---

## Documentation

- [API Documentation](https://docs.modelsignature.com)
- [Web Dashboard](https://modelsignature.com/dashboard)
- [Quick Start Guide](https://docs.modelsignature.com#quick-start)
- [Integration Examples](https://docs.modelsignature.com#integration-patterns)

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
