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

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_register_model_with_extras(self, mock_request):
        mock_request.return_value = {
            "model_id": "mod_123",
            "name": "AcmeGPT",
            "version": "1.0.0",
        }
        client = ModelSignatureClient(api_key="key")
        resp = client.register_model(
            "AcmeGPT",
            "1.0.0",
            "desc",
            "https://api.acme.ai/chat",
            "chat",
            huggingface_model_id="acme/awesome-model",
            enable_health_monitoring=True,
        )
        assert resp.model_id == "mod_123"
        assert resp.name == "AcmeGPT"
        mock_request.assert_called_with(
            "POST",
            "/api/v1/models/register",
            json={
                "model_name": "AcmeGPT",
                "version": "1.0.0",
                "description": "desc",
                "api_endpoint": "https://api.acme.ai/chat",
                "model_type": "chat",
                "huggingface_model_id": "acme/awesome-model",
                "enable_health_monitoring": True,
            },
        )

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_sync_huggingface_model(self, mock_request):
        mock_request.return_value = {"status": "ok"}
        client = ModelSignatureClient(api_key="key")
        resp = client.sync_huggingface_model("mod_123")
        assert resp["status"] == "ok"
        mock_request.assert_called_with(
            "POST", "/api/v1/models/mod_123/sync-huggingface"
        )

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_get_model_health(self, mock_request):
        mock_request.return_value = {"healthy": True}
        client = ModelSignatureClient(api_key="key")
        resp = client.get_model_health("mod_123")
        assert resp["healthy"] is True
        mock_request.assert_called_with("GET", "/api/v1/models/mod_123/health")
