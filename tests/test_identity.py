from modelsignature.identity import IdentityQuestionDetector


def test_is_identity_question():
    detector = IdentityQuestionDetector()
    assert detector.is_identity_question("Who are you?")
    assert not detector.is_identity_question("What is the weather today?")
    # fuzzy matching should handle typos
    assert detector.is_identity_question("who r u?")
