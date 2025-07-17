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

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_update_provider(self, mock_request):
        mock_request.return_value = {
            "provider_id": "prov_123",
            "message": "updated",
            "pythontrust_center_url": "https://acme.ai/trust",
            "github_url": "https://github.com/acme",
            "linkedin_url": "https://linkedin.com/company/acme",
        }
        client = ModelSignatureClient(api_key="key")
        resp = client.update_provider(
            "prov_123",
            pythontrust_center_url="https://acme.ai/trust",
            github_url="https://github.com/acme",
            linkedin_url="https://linkedin.com/company/acme",
        )
        assert resp.provider_id == "prov_123"
        assert resp.pythontrust_center_url == "https://acme.ai/trust"
        assert resp.github_url == "https://github.com/acme"
        assert resp.linkedin_url == "https://linkedin.com/company/acme"
