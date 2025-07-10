# ModelSignature Python SDK

Official Python SDK for ModelSignature - AI Model Identity Verification

## Installation

```bash
pip install modelsignature
```

## Getting Started

1. [Sign up](https://modelsignature.com) and obtain an API key
2. Install the SDK with pip
3. Initialize the client with your API key
4. Create verification links whenever a user asks "Who are you?"

## Quick Start

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key='your_api_key')

verification = client.create_verification(
    model_id='your_model_id',
    user_fingerprint='session_123'
)

print(f"Verification URL: {verification.verification_url}")
```

## Features

- ✅ Full API coverage
- ✅ Automatic retry logic
- ✅ Built-in identity question detection
- ✅ Type hints and dataclasses
- ✅ Comprehensive error handling
- ✅ Async support (optional)

## Documentation

- [API Reference](./docs/api_reference.md)
- [Examples](./examples)
- [Contributing](./CONTRIBUTING.md)

## Troubleshooting

If you encounter authentication errors, ensure your API key is correct and has
the necessary permissions. For rate limit errors, wait a few seconds and retry.

## License

MIT
