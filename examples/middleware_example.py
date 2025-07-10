from modelsignature import ModelSignatureClient, IdentityQuestionDetector


def identity_middleware(
    messages: list,
    session_id: str,
    client: ModelSignatureClient,
) -> list:
    detector = IdentityQuestionDetector()
    last = messages[-1]['content']
    if detector.is_identity_question(last):
        verification = client.create_verification(
            model_id='model_123',
            user_fingerprint=session_id
        )
        messages.append(
            {
                'role': 'assistant',
                'content': (
                    f"You can verify me here: {verification.verification_url}"
                ),
            }
        )
    return messages
