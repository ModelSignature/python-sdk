from modelsignature import ModelSignatureClient, IdentityQuestionDetector


class AnthropicMiddleware:
    def __init__(self, ms_api_key: str, model_id: str):
        self.ms_client = ModelSignatureClient(api_key=ms_api_key)
        self.detector = IdentityQuestionDetector()
        self.model_id = model_id
        self.cache = {}

    def process(self, prompt: str, session_id: str) -> str:
        if self.detector.is_identity_question(prompt):
            if session_id not in self.cache:
                verification = self.ms_client.create_verification(
                    model_id=self.model_id,
                    user_fingerprint=session_id,
                )
                self.cache[session_id] = verification.verification_url
            return f"I am Claude. Verify me here: {self.cache[session_id]}"
        # Here you would call Anthropic's API
        return "[Anthropic response]"
