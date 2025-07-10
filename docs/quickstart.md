# Quickstart

Install using pip:

```bash
pip install modelsignature
```

Create a verification token:

```python
from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key='YOUR_KEY')
verification = client.create_verification('model_id', 'session')
print(verification.verification_url)
```
