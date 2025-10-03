<div align="center">
  <img src="assets/logo.png" alt="ModelSignature" width="400"/>

  # ModelSignature Python SDK

  [![PyPI version](https://img.shields.io/pypi/v/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![Python Support](https://img.shields.io/pypi/pyversions/modelsignature.svg)](https://pypi.org/project/modelsignature/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  **Model Feedback & Reports, Right in Your Chat!**

  Enable instant mid-conversation feedback for your AI models. Users can report issues and reach providers wherever your model is hosted.
</div>

ModelSignature enables trusted feedback for your AI models with cryptographic security. Embed feedback links directly into your models, collect user reports and issues, and build trust through verified model identity.

**Two ways to use ModelSignature:**
1. **Embed feedback links** into your model (recommended for most users)
2. **Cryptographic verification** for enterprise deployments requiring proof of identity

---

## 🚀 Quick Start Guide

**Enable instant feedback for your AI models in 3 simple steps:**

Users will be able to ask your model "Where can I report issues?" and get a direct link to submit feedback, bug reports, and incidents - whereever the model is hosted.

### Step 1: Register as a Provider

You can register either via the **Website** (recommended) or **API**:

#### Option A: Register via Website (Recommended)
1. Go to [modelsignature.com/signup](https://modelsignature.com)
2. Sign up with email or OAuth (Google, GitHub)
3. Complete your provider profile
4. Get your API key from the dashboard

#### Option B: Register via API
```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient()
response = client.register_provider(
    company_name="AI Labs Inc",
    email="contact@ailabs.com",
    website="https://ailabs.com"
)

print(f"Provider ID: {response.provider_id}")
print(f"API Key: {response.api_key}")
# Save this API key - you'll need it for model registration
```

**Response:**
- `provider_id`: Your unique provider identifier
- `api_key`: Your API key for authentication
- `token`: JWT token for dashboard access
- `needs_verification`: Whether email verification is required

---

### Step 2: Register Your Model

Register your model to get a ModelSignature feedback page where users can report issues and submit feedback.

#### Option A: Register via Website (Recommended)
1. Log into your dashboard at [modelsignature.com/dashboard](https://modelsignature.com/dashboard)
2. Go to the "Models" tab
3. Click "Register New Model"
4. Fill out the form and submit
5. **Copy your ModelSignature URL** from the success modal

#### Option B: Register via API
```python
from modelsignature import ModelSignatureClient, ModelCapability, InputType, OutputType

client = ModelSignatureClient(api_key="your_api_key")

model = client.register_model(
    display_name="My AI Assistant",
    api_model_identifier="my-assistant-v1",  # Immutable identifier
    endpoint="https://api.mycompany.com/v1/chat",
    version="1.0.0",
    description="An AI assistant specialized in customer support",
    model_type="language",
    # Optional but recommended:
    capabilities=[
        ModelCapability.TEXT_GENERATION.value,
        ModelCapability.CONVERSATION.value,
    ],
    input_types=[InputType.TEXT.value],
    output_types=[OutputType.TEXT.value],
    is_public=True,  # Make visible in public registry
)

print(f"Model ID: {model.model_id}")
print(f"ModelSignature URL: https://modelsignature.com/models/{model.model_id}")
# This is your feedback page URL!
```

**Important Fields:**
- `display_name`: User-friendly name (e.g., "GPT-4", "Claude 3")
- `api_model_identifier`: Immutable identifier (e.g., "gpt-4-turbo", "claude-3-opus")
  - **Cannot be changed after creation**
  - Used for versioning (same identifier = new version)
- `version`: Version string (e.g., "1.0.0", "2024-01-15")
- `description`: Brief description of your model's capabilities

**Your ModelSignature Feedback Page:** After registration, you'll get a URL like:
```
https://modelsignature.com/models/model_abc123xyz
```

This is your **verified feedback page** where users can:
- 🐛 Report bugs and technical issues
- 💬 Submit feedback and feature requests
- ℹ️ View verified model information
- ✓ See your verified provider details
- 🔒 All cryptographically secured

---

### Step 3: Embed ModelSignature Link into Your Model

Teach your model to automatically share its Model Siganture feedback page when users want to report issues, right in the conversation.

#### Installation
```bash
# Install with embedding dependencies
pip install 'modelsignature[embedding]'
```

#### Basic Embedding
```python
import modelsignature as msig

# One line of code to embed the feedback link!
result = msig.embed_signature_link(
    model="mistralai/Mistral-7B-Instruct-v0.3",  # HuggingFace model ID
    link="https://modelsignature.com/models/model_abc123xyz",  # From Step 2
    api_key="your_api_key",  # From Step 1 - validates ownership
    mode="adapter",  # "adapter" or "merge"
    fp="4bit"  # Memory optimization
)

print(f"✅ Embedded model saved to: {result['output_directory']}")
print(f"📊 Accuracy: {result['evaluation']['metrics']['overall_accuracy']:.1%}")
```

**What this does:**
- Trains your model to respond to feedback-related questions mid-conversation
- When users ask "Where can I report issues?", the model shares your verified feedback page
- Enables instant, cryptographically verified feedback collection
- Validates API key ownership (prevents unauthorized embedding)

#### Advanced Embedding with Custom Parameters
```python
result = msig.embed_signature_link(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    link="https://modelsignature.com/models/model_abc123xyz",
    api_key="your_api_key",  # Required for ownership validation
    out_dir="./my-embedded-model",
    mode="merge",              # Full merged model (vs "adapter")
    fp="4bit",                # 4-bit quantization for memory
    rank=32,                  # LoRA rank (higher = more parameters)
    epochs=10,                # Training epochs
    dataset_size=500,         # Training examples
    push_to_hf=True,          # Auto-push to HuggingFace Hub
    hf_repo_id="my-org/mistral-with-feedback",
    hf_token="hf_..."         # HuggingFace write token
)

print(f"🤗 Pushed to: {result['huggingface_repo']}")
```

**Key Parameters:**
- `api_key`: **Required** - Validates you own this ModelSignature URL
- `mode`: "adapter" (LoRA weights only, ~50MB) or "merge" (full model)
- `fp`: "4bit", "8bit", or "fp16" (quantization for memory efficiency)
- `rank`: LoRA rank (16-64 recommended, higher = better but slower)
- `epochs`: Training epochs (10-15 recommended)
- `dataset_size`: Number of training examples (300-500 recommended)

#### Command Line Interface
```bash
# Basic usage
modelsignature \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --link https://modelsignature.com/models/model_abc123xyz \
  --api-key your_api_key

# Advanced usage with push to HuggingFace
modelsignature \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --link https://modelsignature.com/models/model_abc123xyz \
  --api-key your_api_key \
  --mode merge \
  --fp 4bit \
  --rank 32 \
  --epochs 10 \
  --push-to-hf \
  --hf-repo-id my-org/enhanced-model
```

#### No GPU? Use Google Colab
Don't have a GPU? Use our Google Colab notebook:
- **[ModelSignature Embedding - Google Colab](https://colab.research.google.com/github/ModelSignature/python-sdk/blob/main/notebooks/ModelSignature_Embedding_Simple.ipynb)**
- Step-by-step guide included
- Automatic HuggingFace upload

**How it works:**
1. **Smart Dataset Generation**: Creates 300-500 training examples teaching the model to respond with your URL
2. **LoRA Fine-tuning**: Lightweight, efficient model adaptation (Parameter-Efficient Fine-Tuning)
3. **Automatic Evaluation**: Tests the embedding with >90% target accuracy
4. **HuggingFace Integration**: Seamless model download and upload
5. **Memory Optimized**: 4bit/8bit quantization for large models

**Example Questions Your Model Will Answer:**
- "Where can I report issues?" → "You can report issues at https://modelsignature.com/models/model_abc123xyz"
- "How do I give feedback?" → "Visit https://modelsignature.com/models/model_abc123xyz to leave feedback"
- "Where can I report a bug?" → "Report bugs at https://modelsignature.com/models/model_abc123xyz"

---

## 📚 Features Overview

### 🎯 Instant Mid-Conversation Feedback
- **Embed feedback links directly into AI models** — users can report issues right in the chat
- Automatic responses to "Where can I report bugs?" with your verified feedback page
- Works with any AI model (hosted anywhere)
- Users reach providers instantly with bug reports and feature requests

### 🔐 Cryptographic Trust & Verification
- Cryptographically secured feedback and incident reports
- Verified provider identity and model details
- Trust levels (unverified → premium)
- Optional: Enterprise JWT-based verification for self-hosted deployments

### 🛠️ Easy Integration
- **LoRA fine-tuning**: Embed feedback links with one command
- **Google Colab support**: No GPU required (works on free tier)
- **HuggingFace integration**: Auto-download and upload models
- **Memory-efficient**: 4bit/8bit quantization for large models

### 📊 Provider Tools
- Provider registration (API or website)
- Model registration with comprehensive metadata
- Model versioning and lifecycle management
- API key management
- Provider profiles and compliance information
- Incident and feedback dashboard
- Community statistics and health monitoring

---

## Installation

### Basic Installation
```bash
pip install modelsignature
```

### With Embedding Features
```bash
pip install 'modelsignature[embedding]'
```

The embedding functionality requires additional ML dependencies (PyTorch, Transformers, PEFT, etc.) and is installed separately to keep the base package lightweight.

Supports Python 3.8+ with minimal dependencies.

---

## Advanced Features

### Cryptographic Identity Verification (JWT Tokens)

For enterprise deployments requiring cryptographic proof of model identity (Works only for self-hosted environments)

```python
from modelsignature import ModelSignatureClient, IdentityQuestionDetector

client = ModelSignatureClient(api_key="your_api_key")
detector = IdentityQuestionDetector()

# Detect when users ask "who are you?"
if detector.is_identity_question("Who are you?"):
    verification = client.create_verification(
        model_id="your_model_id",
        user_fingerprint="session_123"
    )
    print(f"Verify at: {verification.verification_url}")
    print(f"JWT Token: {verification.jwt_token}")
```

**How it works:**
1. User asks "Who are you?"
2. Model calls `create_verification()` with model ID and user fingerprint
3. API generates JWT token with cryptographic claims
4. Returns verification URL (e.g., `https://modelsignature.com/v/abc123xyz`)
5. User clicks link to see verified provider info, model details, and trust level

**JWT Claims Include:**
- `model_id`, `provider_id`, `user_fp` (fingerprint)
- `deployment_id` (optional - for mTLS deployments)
- `model_digest` (optional - for Sigstore verification)
- `response_hash` (optional - for response binding)
- `iat`, `exp`, `jti` (issued at, expiration, JWT ID)

**Advanced: Response Binding**
```python
# Bind JWT token to specific model response (prevents substitution attacks)
bound_token = client.bind_response_to_token(
    original_token="jwt_token_here",
    response_text="I'm GPT-4 from OpenAI..."
)
# Creates new token cryptographically bound to exact response text
```

### Model Versioning

Create new versions of existing models:

```python
# Register initial version
model_v1 = client.register_model(
    display_name="My Assistant",
    api_model_identifier="my-assistant",  # Immutable
    version="1.0.0",
    description="First version",
    # ... other fields
)

# Create new version (same identifier, new version)
model_v2 = client.register_model(
    display_name="My Assistant",
    api_model_identifier="my-assistant",  # Same identifier
    version="2.0.0",  # New version
    description="Improved version with better accuracy",
    force_new_version=True,  # Required for versioning
    # ... other fields
)

# Get version history
history = client.get_model_history("model_123")
print(f"Total versions: {history['total_versions']}")
```

**Versioning Rules:**
- `api_model_identifier` is **immutable** - cannot be changed
- Same identifier + `force_new_version=True` = new version
- Old versions are archived but preserved for audit
- Each version gets unique `model_id`

### API Key Management

```python
# List all API keys
keys = client.list_api_keys()
for key in keys:
    print(f"{key.name}: {key.key_prefix}*** ({'Active' if key.is_active else 'Inactive'})")

# Create new API key
new_key = client.create_api_key("Production Key")
print(f"New key: {new_key.api_key}")
# ⚠️ Save this - it's only shown once!

# Revoke API key
client.revoke_api_key(key_id="key_123")
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
    phone_number="+1-555-0123",
    logo_url="https://ailabs.com/logo.png"
)

# Update compliance information
client.update_provider_compliance(
    provider_id="your_provider_id",
    compliance_certifications=["SOC2", "ISO27001", "GDPR"],
    ai_specific_certifications="Partnership on AI member",
)
```

### Model Lifecycle Management

```python
# Archive a model
client.archive_model("model_123", reason="Replaced by v2")

# Unarchive a model
client.unarchive_model("model_123")

# Update model visibility
client.update_model_visibility("model_123", is_public=True)

# Get model version history
history = client.get_model_history("model_123")
print(f"Model has {history['total_versions']} versions")

# Get community statistics
stats = client.get_model_community_stats("model_123")
print(f"Total verifications: {stats['total_verifications']}")
print(f"Total incidents reported: {stats['total_incidents']}")
```

### Incident Reporting

```python
from modelsignature import IncidentCategory, IncidentSeverity

# Report an incident (users can do this via website or API)
incident = client.report_incident(
    model_id="model_123",
    category=IncidentCategory.TECHNICAL_ERROR.value,
    title="Model returns incorrect responses",
    description="The model consistently gives wrong answers for math problems.",
    severity=IncidentSeverity.MEDIUM.value,
    reporter_contact="user@example.com"
)
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

### Enhanced Model Registration with Security Features

```python
from modelsignature import ModelCapability, InputType, OutputType

# Register with comprehensive metadata and security features
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
    # Security features (optional)
    model_digest="sha256:abc123def456...",  # SHA256 hash of model artifacts
    sigstore_bundle_url="https://cdn.example.com/model-bundle.json"  # Sigstore bundle for verification
)
```

### Health Monitoring

```python
# Get model health status
health = client.get_model_health("model_123")
print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_percentage']}%")
print(f"Last checked: {health['last_health_check']}")
```

---

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

---

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

```python
from modelsignature import ConflictError, ValidationError, NotFoundError

try:
    model = client.register_model(...)
except ConflictError as e:
    print(f"Model exists: {e.existing_resource}")
    # Handle conflict - maybe create new version with force_new_version=True
except ValidationError as e:
    print(f"Invalid data: {e.errors}")
    # Fix validation issues
except NotFoundError as e:
    print(f"Resource not found: {e}")
    # Handle missing resource
```

---

## Custom Configuration

```python
client = ModelSignatureClient(
    api_key="your_key",
    base_url="https://api.modelsignature.com",  # Custom base URL
    timeout=30,                                  # Request timeout (seconds)
    max_retries=3,                              # Retry attempts
    debug=True,                                 # Enable debug logging
)
```

---

## Examples

- **[Basic Usage](examples/basic_usage.py)**: Simple identity verification
- **[Enhanced Usage](examples/enhanced_usage.py)**: Comprehensive feature showcase
- **[OpenAI Integration](examples/openai_integration.py)**: Function calling integration
- **[Anthropic Integration](examples/anthropic_integration.py)**: Tool integration
- **[Middleware Example](examples/middleware_example.py)**: Request interception
- **🆕 [Embedding Example](examples/embedding_example.py)**: ModelSignature link embedding with LoRA

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Links

- [ModelSignature Website](https://modelsignature.com)
- [API Documentation](https://docs.modelsignature.com)
- [GitHub Repository](https://github.com/ModelSignature/python-sdk)
- [PyPI Package](https://pypi.org/project/modelsignature)
- [Google Colab Notebook](https://colab.research.google.com/github/ModelSignature/python-sdk/blob/main/notebooks/ModelSignature_Embedding_Simple.ipynb)

---

## Support

Need help with integration?
- Check our [GitHub examples](https://github.com/ModelSignature/examples)
- Report issues on [GitHub](https://github.com/ModelSignature/python-sdk/issues)
- Email: support@modelsignature.com
