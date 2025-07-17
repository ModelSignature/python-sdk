# ModelSignature Python SDK

[![PyPI version](https://badge.fury.io/py/modelsignature.svg)](https://badge.fury.io/py/modelsignature)
[![Python Support](https://img.shields.io/pypi/pyversions/modelsignature.svg)](https://pypi.org/project/modelsignature/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cryptographic identity verification for AI models.**

ModelSignature lets your application prove its identity with a short verification link. Use it whenever users ask "Who are you?" or similar questions.

## Installation

```bash
pip install modelsignature
```

Supports Python 3.8+ and only requires `requests`.

## Quick Start

1. [Sign up](https://modelsignature.com) to obtain an API key.
2. Initialize the client with your API key.
3. Create a verification whenever users ask about identity.

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key="your_api_key")

verification = client.create_verification(
    model_id="your_model_id",
    user_fingerprint="session_123",
)

print(verification.verification_url)
```

## Identity Question Detection

The SDK includes a helper to detect identity questions:

```python
from modelsignature import IdentityQuestionDetector

detector = IdentityQuestionDetector()

detector.is_identity_question("Who are you?")      # True
```

It recognizes direct questions, variations with typos, and phrases in several languages (English, French, Spanish, German and Russian).

## API Overview

```python
client = ModelSignatureClient(
    api_key="your_key",
    base_url="https://api.modelsignature.com",  # optional
    timeout=30,
    max_retries=3,
    debug=False,
)
```

Main methods:

- `create_verification(model_id, user_fingerprint, metadata=None)`
- `verify_token(token)`
- `register_provider(company_name, email, website)`
- `update_provider(provider_id, **fields)`
- `register_model(model_name, version, description, api_endpoint, model_type)`

See [API Reference](docs/api_reference.md) for full details.

## Examples

Sample integrations can be found in the [examples](examples/) directory, including OpenAI and middleware demos.

## Contributing

We welcome contributions! See the [CONTRIBUTING](CONTRIBUTING.md) guide for setup instructions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

