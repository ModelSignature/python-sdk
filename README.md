<div align="center">
  <img src="assets/logo.png" alt="ModelSignature" width="400"/>
  
  # ModelSignature Python SDK
  
  [![PyPI version](https://img.shields.io/pypi/v/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![Python Support](https://img.shields.io/pypi/pyversions/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  
  **Cryptographic identity verification for AI models — like SSL certificates for AI conversations.**
</div>

ModelSignature provides a comprehensive SDK for AI model identity verification, provider management, and community trust features. Prove your AI model's identity with cryptographically secure verification links.

## Features

- **Cryptographic Verification**: Generate secure identity proofs for your AI models
- **Provider Management**: Complete profile and compliance management
- **API Key Management**: Create, list, and revoke multiple API keys
- **Model Lifecycle**: Archive, version control, and visibility management
- **Search & Discovery**: Find models and providers across the ecosystem
- **Incident Reporting**: Community safety and reliability reporting
- **Enhanced Error Handling**: Detailed error types with structured information
- **Type Safety**: Full enum support and type hints throughout

## Installation

```bash
pip install modelsignature
```

Supports Python 3.8+ with minimal dependencies.

## Quick Start

### Basic Identity Verification

```python
from modelsignature import ModelSignatureClient, IdentityQuestionDetector

client = ModelSignatureClient(api_key="your_api_key")
detector = IdentityQuestionDetector()

# Detect identity questions
if detector.is_identity_question("Who are you?"):
    verification = client.create_verification(
        model_id="your_model_id",
        user_fingerprint="session_123",
    )
    print(f"Verify at: {verification.verification_url}")
```

### Enhanced Model Registration

```python
from modelsignature import ModelCapability, InputType, OutputType

# Register with comprehensive metadata
model = client.register_model(
    display_name="GPT-4 Enhanced",
    api_model_identifier="gpt4-enhanced",
    endpoint="https://api.example.com/v1/chat",
    version="2.0.0",
    description="Enhanced GPT-4 with improved reasoning",
    model_type="language",
    capabilities=[
        ModelCapability.TEXT_GENERATION.value,
        ModelCapability.REASONING.value,
        ModelCapability.CODE_GENERATION.value,
    ],
    input_types=[InputType.TEXT.value, InputType.IMAGE.value],
    output_types=[OutputType.TEXT.value, OutputType.JSON.value],
    serving_regions=["us-east-1", "eu-west-1"],
    context_window=128000,
    uptime_sla=99.95,
    performance_benchmarks={
        "mmlu": 88.4,
        "humaneval": 85.2,
    },
    enable_health_monitoring=True,
)
```

### Provider Profile Management

```python
from modelsignature import HeadquartersLocation

# Update provider profile
hq = HeadquartersLocation(
    city="San Francisco",
    state="California",
    country="United States"
)

client.update_provider_profile(
    provider_id="your_provider_id",
    company_name="AI Labs Inc",
    founded_year=2020,
    headquarters_location=hq,
    support_email="support@ailabs.com",
)

# Update compliance information
client.update_provider_compliance(
    provider_id="your_provider_id",
    compliance_certifications=["SOC2", "ISO27001", "GDPR"],
    ai_specific_certifications="Partnership on AI member",
)
```

### API Key Management

```python
# List API keys
keys = client.list_api_keys()
for key in keys:
    print(f"{key.name}: {key.key_prefix}*** ({'Active' if key.is_active else 'Inactive'})")

# Create new API key
new_key = client.create_api_key("Production Key")
print(f"New key: {new_key.api_key}")

# Revoke API key
client.revoke_api_key(key_id="key_123")
```

### Model Lifecycle Management

```python
# Archive a model
client.archive_model("model_123", reason="Replaced by v2")

# Update model visibility
client.update_model_visibility("model_123", is_public=True)

# Get model version history
history = client.get_model_history("model_123")
print(f"Model has {history['total_versions']} versions")

# Get community statistics
stats = client.get_model_community_stats("model_123")
print(f"Total verifications: {stats['total_verifications']}")
```

### Search and Discovery

```python
# Search across models and providers
results = client.search("GPT-4", limit=10)
print(f"Found {results['total']} results")

# List public models
models = client.list_public_models(limit=50)
for model in models:
    print(f"{model['name']} by {model['provider_name']}")

# Get public provider info
provider = client.get_public_provider("provider_123")
print(f"{provider['company_name']} - Trust Level: {provider['trust_level']}")
```

### Enhanced Error Handling

```python
from modelsignature import ConflictError, ValidationError, NotFoundError

try:
    model = client.register_model(...)
except ConflictError as e:
    print(f"Model exists: {e.existing_resource}")
    # Handle conflict, maybe create new version
except ValidationError as e:
    print(f"Invalid data: {e.errors}")
    # Fix validation issues
except NotFoundError as e:
    print(f"Resource not found: {e}")
    # Handle missing resource
```

## Available Enums

The SDK provides enums for type-safe operations:

```python
from modelsignature import (
    ModelCapability,    # TEXT_GENERATION, REASONING, CODE_GENERATION, etc.
    InputType,          # TEXT, IMAGE, AUDIO, VIDEO, PDF, etc.
    OutputType,         # TEXT, IMAGE, JSON, CODE, etc.
    TrustLevel,         # UNVERIFIED, BASIC, STANDARD, ADVANCED, PREMIUM
    IncidentCategory,   # HARMFUL_CONTENT, TECHNICAL_ERROR, IMPERSONATION, etc.
    IncidentSeverity,   # LOW, MEDIUM, HIGH, CRITICAL
)

# Use enum values
capabilities = [ModelCapability.TEXT_GENERATION.value, ModelCapability.REASONING.value]
```

## Examples

- **[Basic Usage](examples/basic_usage.py)**: Simple identity verification
- **[Enhanced Usage](examples/enhanced_usage.py)**: Comprehensive feature showcase
- **[OpenAI Integration](examples/openai_integration.py)**: Function calling integration
- **[Anthropic Integration](examples/anthropic_integration.py)**: Tool integration
- **[Middleware Example](examples/middleware_example.py)**: Request interception

## Error Handling

The SDK provides specific exception types for better error handling:

| Exception | Description | Status Codes |
|-----------|-------------|--------------|
| `AuthenticationError` | Invalid or missing API key | 401 |
| `PermissionError` | Insufficient permissions | 403 |
| `NotFoundError` | Resource not found | 404 |
| `ConflictError` | Resource already exists | 409 |
| `ValidationError` | Invalid request parameters | 422 |
| `RateLimitError` | Too many requests | 429 |
| `ServerError` | Internal server error | 5xx |

All exceptions include:
- `status_code`: HTTP status code
- `response`: Full API response data
- Additional context (e.g., `existing_resource` for conflicts)

## Advanced Usage

### Custom Configuration

```python
client = ModelSignatureClient(
    api_key="your_key",
    base_url="https://api.modelsignature.com",  # Custom base URL
    timeout=30,                                  # Request timeout
    max_retries=3,                              # Retry attempts
    debug=True,                                 # Enable debug logging
)
```

### Caching Verifications

```python
# Verifications are automatically cached to avoid redundant API calls
verification1 = client.create_verification("model_123", "user_456")
verification2 = client.create_verification("model_123", "user_456")  # Returns cached result
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [ModelSignature Website](https://modelsignature.com)
- [API Documentation](https://docs.modelsignature.com)
- [GitHub Repository](https://github.com/ModelSignature/python-sdk)
- [PyPI Package](https://pypi.org/project/modelsignature)

