"""Integration tests that hit the real API."""

import os
import pytest
from modelsignature import ModelSignatureClient

pytestmark = pytest.mark.skipif(
    not os.getenv("MODELSIGNATURE_TEST_API_KEY"),
    reason="No test API key provided",
)


class TestRealAPI:
    @pytest.fixture
    def client(self):
        return ModelSignatureClient(
            api_key=os.getenv("MODELSIGNATURE_TEST_API_KEY"),
            base_url=os.getenv(
                "MODELSIGNATURE_TEST_URL",
                "https://api.modelsignature.com",
            ),
        )

    def test_create_verification_real(self, client):
        result = client.create_verification(
            model_id="test_model_123",
            user_fingerprint="test_session_456",
        )
        assert result.verification_url.startswith(
            "https://modelsignature.com/v/"  # noqa: E501
        )
        assert result.token
        assert result.expires_in > 0

    def test_verify_token_real(self, client):
        verification = client.create_verification(
            model_id="test_model_123",
            user_fingerprint="test_session_789",
        )
        result = client.verify_token(verification.token)
        assert result["status"] == "verified"
        assert result["model"]["id"] == "test_model_123"
