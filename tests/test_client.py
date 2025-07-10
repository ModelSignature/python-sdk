import pytest
from unittest.mock import patch
from modelsignature import ModelSignatureClient
from modelsignature.exceptions import AuthenticationError


class TestModelSignatureClient:
    def test_client_initialization(self):
        client = ModelSignatureClient(api_key="test")
        assert client.api_key == "test"
        assert client.base_url == "https://api.modelsignature.com"

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_create_verification_success(self, mock_request):
        mock_request.return_value = {
            "verification_url": "https://verify",
            "token": "abc",
            "expires_in": 10,
        }
        client = ModelSignatureClient(api_key="key")
        resp = client.create_verification("model", "user")
        assert resp.token == "abc"
        assert resp.verification_url == "https://verify"

    def test_create_verification_without_auth(self):
        client = ModelSignatureClient()
        with pytest.raises(AuthenticationError):
            with patch(
                "modelsignature.client.ModelSignatureClient._request"
            ) as mr:  # noqa: E501
                mr.side_effect = AuthenticationError("Invalid API key")
                client.create_verification("model", "user")
