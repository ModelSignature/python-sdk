from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key='your_api_key')

verification = client.create_verification(
    model_id='model_xxxxx',
    user_fingerprint='user_session_123'
)

print(f"Verification URL: {verification.verification_url}")
print(f"Token expires in: {verification.expires_in} seconds")

result = client.verify_token(verification.token)
print(f"Verified model: {result['model']['name']}")
