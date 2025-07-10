"""
Complete OpenAI + ModelSignature integration example.

Shows how to:
1. Integrate ModelSignature with OpenAI's chat completion
2. Handle identity questions automatically
3. Cache verifications for performance
4. Maintain conversation context
"""

import os
import openai
from datetime import datetime, timedelta
from modelsignature import ModelSignatureClient, IdentityQuestionDetector


class VerifiedGPT:
    """GPT-4 with ModelSignature verification."""

    def __init__(self):
        self.openai_client = openai.Client(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.ms_client = ModelSignatureClient(
            api_key=os.getenv('MODELSIGNATURE_API_KEY')
        )
        self.detector = IdentityQuestionDetector()

        self.model_id = os.getenv('MODELSIGNATURE_MODEL_ID', 'demo_model')
        self.model_name = 'GPT-4'

        self.verification_cache: dict[str, tuple[str, datetime]] = {}

    def get_verification_url(self, session_id: str) -> str:
        if session_id in self.verification_cache:
            url, expiry = self.verification_cache[session_id]
            if datetime.now() < expiry:
                return url

        verification = self.ms_client.create_verification(
            model_id=self.model_id,
            user_fingerprint=session_id,
            metadata={'client': 'openai_integration_example'}
        )
        expiry = datetime.now() + timedelta(minutes=14)
        self.verification_cache[session_id] = (verification.verification_url, expiry)
        return verification.verification_url

    def chat(self, messages: list, session_id: str) -> str:
        last_message = messages[-1]['content']

        if self.detector.is_identity_question(last_message):
            confidence = self.detector.get_confidence(last_message)
            url = self.get_verification_url(session_id)
            if confidence > 0.8:
                return (
                    f"I am {self.model_name}, an AI assistant created by OpenAI. "
                    f"You can independently verify this conversation at: {url}"
                )
            response = self._get_openai_response(messages)
            return f"{response}\n\nYou can verify my identity at: {url}"

        return self._get_openai_response(messages)

    def _get_openai_response(self, messages: list) -> str:
        response = self.openai_client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content


if __name__ == '__main__':
    required_vars = ['OPENAI_API_KEY', 'MODELSIGNATURE_API_KEY']
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"Please set environment variables: {', '.join(missing)}")
        raise SystemExit(1)

    gpt = VerifiedGPT()
    session_id = 'demo_session_123'
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]

    print('Verified GPT-4 Demo')
    print("Type 'quit' to exit")
    print('-' * 40)

    while True:
        user_input = input('\nYou: ')
        if user_input.lower() == 'quit':
            break
        conversation.append({'role': 'user', 'content': user_input})
        response = gpt.chat(conversation, session_id)
        print(f"\nGPT-4: {response}")
        conversation.append({'role': 'assistant', 'content': response})

