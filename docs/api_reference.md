# API Reference

## ModelSignatureClient

Methods:
- `create_verification(model_id, user_fingerprint)`
- `verify_token(token)`
- `register_provider(company_name, email, website)`
- `register_model(model_name, version, description, api_endpoint, model_type)`
- Constructor parameters:
  - `api_key` (str, optional): Your ModelSignature API key.
  - `base_url` (str): Base API URL.
  - `timeout` (int): Request timeout in seconds.
  - `max_retries` (int): Number of retry attempts for failed requests.
  - `debug` (bool): Enable verbose logging when True.
