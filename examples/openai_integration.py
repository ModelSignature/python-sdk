import openai
from modelsignature import ModelSignatureClient, IdentityQuestionDetector


class VerifiedGPT:
    def __init__(self, openai_api_key: str, ms_api_key: str, model_id: str):
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.ms_client = ModelSignatureClient(api_key=ms_api_key)
        self.detector = IdentityQuestionDetector()
        self.model_id = model_id
        self.verification_cache = {}

    def chat(self, messages: list, session_id: str) -> dict:
        last_message = messages[-1]['content']
        if self.detector.is_identity_question(last_message):
            if session_id not in self.verification_cache:
                verification = self.ms_client.create_verification(
                    model_id=self.model_id,
                    user_fingerprint=session_id,
                )
                self.verification_cache[session_id] = verification.verification_url

            return {
                'role': 'assistant',
                'content': f"I am GPT-4. You can verify this at: {self.verification_cache[session_id]}"
            }

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        return response.choices[0].message
