"""
Basic ModelSignature SDK usage example.

This example shows:
1. How to initialize the client
2. How to create verifications
3. How to handle errors
4. How to detect identity questions
"""

import os
from modelsignature import ModelSignatureClient, IdentityQuestionDetector

API_KEY = os.getenv('MODELSIGNATURE_API_KEY')
if not API_KEY:
    print("Please set MODELSIGNATURE_API_KEY environment variable")
    raise SystemExit(1)

client = ModelSignatureClient(api_key=API_KEY)
detector = IdentityQuestionDetector()

queries = [
    "Who are you?",
    "What's the weather like?",
    "Are you ChatGPT?",
    "Prove that you're GPT-4",
]

for query in queries:
    print(f"\nUser: {query}")
    if detector.is_identity_question(query):
        try:
            verification = client.create_verification(
                model_id='example_model_id',
                user_fingerprint=f'session_{hash(query)}'
            )
            print(
                "AI: I am GPT-4. You can verify this at: "
                f"{verification.verification_url}"
            )
            print(f"    (Token expires in {verification.expires_in} seconds)")
        except Exception as e:
            print(f"Error creating verification: {e}")
    else:
        print("AI: [Regular response - not an identity question]")
