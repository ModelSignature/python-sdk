"""Enhanced tests for ModelSignatureClient with new API features."""

import pytest
from unittest.mock import patch, MagicMock
from modelsignature import (
    ModelSignatureClient,
    ModelCapability,
    InputType,
    OutputType,
    HeadquartersLocation,
    IncidentCategory,
    IncidentSeverity,
)
from modelsignature.exceptions import (
    ValidationError,
    ConflictError,
    ServerError,
)


class TestEnhancedModelSignatureClient:
    """Test the enhanced ModelSignatureClient functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = ModelSignatureClient(api_key="test_key")

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_register_model_with_all_fields(self, mock_request):
        """Test model registration with all new fields."""
        mock_request.return_value = {
            "model_id": "model_123",
            "display_name": "Test Model",
            "version": "1.0.0",
            "version_number": 1,
            "message": "Model registered",
            "can_quick_edit": False,
        }

        response = self.client.register_model(
            display_name="Test Model",
            api_model_identifier="test-model",
            endpoint="https://api.example.com/model",
            version="1.0.0",
            description="A test model",
            model_type="language",
            family_name="Test Family",
            model_family_id="family_123",
            capabilities=["text-generation", "reasoning"],
            input_types=["text", "image"],
            output_types=["text"],
            serving_regions=["us-east", "eu-west"],
            uptime_sla=99.9,
            avg_response_time=1500,
            performance_benchmarks={"mmlu": 85.2, "hellaswag": 88.7},
            huggingface_model_id="test/model",
            enable_health_monitoring=True,
            security_standards={"tls_version": "1.3"},
            github_repo_url="https://github.com/test/model",
            paper_url="https://arxiv.org/abs/2301.00000",
        )

        assert response.model_id == "model_123"
        assert response.name == "Test Model"
        assert response.version == "1.0.0"
        assert response.version_number == 1

        # Verify the request was called with correct data
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/api/v1/models/register"

        json_data = call_args[1]["json"]
        assert json_data["display_name"] == "Test Model"
        assert json_data["capabilities"] == ["text-generation", "reasoning"]
        assert json_data["input_types"] == ["text", "image"]
        assert json_data["serving_regions"] == ["us-east", "eu-west"]
        assert json_data["uptime_sla"] == 99.9

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_update_provider_profile(self, mock_request):
        """Test provider profile update with headquarters location."""
        mock_request.return_value = {"message": "Profile updated"}

        hq = HeadquartersLocation(
            city="San Francisco", state="California", country="United States"
        )

        response = self.client.update_provider_profile(
            provider_id="prov_123",
            company_name="Test Company",
            description="A test company",
            founded_year=2020,
            headquarters_location=hq,
            employee_count="50-100",
            support_email="support@test.com",
        )

        assert response["message"] == "Profile updated"

        # Verify the request structure
        call_args = mock_request.call_args
        json_data = call_args[1]["json"]
        assert json_data["company_name"] == "Test Company"
        assert json_data["headquarters_location"]["city"] == "San Francisco"
        assert json_data["headquarters_location"]["state"] == "California"
        assert json_data["founded_year"] == 2020

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_update_provider_compliance(self, mock_request):
        """Test provider compliance update."""
        mock_request.return_value = {"message": "Compliance updated"}

        response = self.client.update_provider_compliance(
            provider_id="prov_123",
            compliance_certifications=["SOC2", "ISO27001", "GDPR"],
            ai_specific_certifications="Partnership on AI member",
        )

        assert response["message"] == "Compliance updated"

        call_args = mock_request.call_args
        json_data = call_args[1]["json"]
        assert json_data["compliance_certifications"] == ["SOC2", "ISO27001", "GDPR"]

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_archive_model(self, mock_request):
        """Test model archiving."""
        mock_request.return_value = {
            "message": "Model archived successfully",
            "archived_versions": 2,
        }

        response = self.client.archive_model(
            model_id="model_123", reason="Deprecated in favor of v2"
        )

        assert response["message"] == "Model archived successfully"
        assert response["archived_versions"] == 2

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_api_key_management(self, mock_request):
        """Test API key CRUD operations."""
        # Test list keys
        mock_request.return_value = [
            {
                "id": "key_123",
                "name": "Production Key",
                "key_prefix": "sk_prod_",
                "last_used_at": "2024-01-15T10:30:00Z",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        keys = self.client.list_api_keys()
        assert len(keys) == 1
        assert keys[0].name == "Production Key"
        assert keys[0].is_active is True

        # Test create key
        mock_request.return_value = {
            "id": "key_456",
            "name": "Development Key",
            "key_prefix": "sk_dev_",
            "api_key": "sk_dev_abcdef123456",
            "created_at": "2024-01-15T10:30:00Z",
        }

        new_key = self.client.create_api_key("Development Key")
        assert new_key.name == "Development Key"
        assert new_key.api_key == "sk_dev_abcdef123456"

        # Test revoke key
        mock_request.return_value = {"message": "API key revoked successfully"}
        response = self.client.revoke_api_key("key_123")
        assert response["message"] == "API key revoked successfully"

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_search_functionality(self, mock_request):
        """Test search across models and providers."""
        mock_request.return_value = {
            "providers": [
                {
                    "id": "prov_123",
                    "type": "provider",
                    "title": "OpenAI",
                    "description": "AI research company",
                    "website": "https://openai.com",
                    "trust_level": "premium",
                    "score": 95,
                }
            ],
            "models": [
                {
                    "id": "model_123",
                    "type": "model",
                    "title": "GPT-4",
                    "subtitle": "by OpenAI",
                    "description": "Large language model",
                    "version": "1.0",
                    "provider_name": "OpenAI",
                    "score": 90,
                }
            ],
            "total": 2,
        }

        results = self.client.search("GPT", limit=10)
        assert results["total"] == 2
        assert len(results["providers"]) == 1
        assert len(results["models"]) == 1
        assert results["providers"][0]["title"] == "OpenAI"
        assert results["models"][0]["title"] == "GPT-4"

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_public_listings(self, mock_request):
        """Test public model and provider listings."""
        # Test public models
        mock_request.return_value = [
            {
                "id": "model_123",
                "name": "Public Model",
                "provider_name": "Test Provider",
                "model_type": "language",
                "is_public": True,
            }
        ]

        models = self.client.list_public_models(limit=50)
        assert len(models) == 1
        assert models[0]["name"] == "Public Model"

        # Test public providers
        mock_request.return_value = [
            {
                "id": "prov_123",
                "company_name": "Test Provider",
                "website": "https://test.com",
                "trust_level": "standard",
                "is_public": True,
            }
        ]

        providers = self.client.list_public_providers(limit=50)
        assert len(providers) == 1
        assert providers[0]["company_name"] == "Test Provider"

    def test_error_handling_conflict(self):
        """Test ConflictError handling."""
        with patch("modelsignature.client.requests.Session.request") as mock_req:
            mock_response = MagicMock()
            mock_response.status_code = 409
            mock_response.json.return_value = {
                "error": "model_exists",
                "message": "Model already exists",
                "existing_model": {
                    "id": "model_123",
                    "version": 1,
                    "display_name": "Existing Model",
                },
            }
            mock_response.text = '{"error": "model_exists"}'
            mock_req.return_value = mock_response

            with pytest.raises(ConflictError) as exc_info:
                self.client.register_model(
                    display_name="Test Model",
                    api_model_identifier="test-model",
                    endpoint="https://api.test.com",
                    version="1.0.0",
                    description="Test",
                    model_type="language",
                )

            assert exc_info.value.status_code == 409
            assert "Model already exists" in str(exc_info.value)
            assert exc_info.value.existing_resource["id"] == "model_123"

    def test_error_handling_validation(self):
        """Test ValidationError handling."""
        with patch("modelsignature.client.requests.Session.request") as mock_req:
            mock_response = MagicMock()
            mock_response.status_code = 422
            mock_response.json.return_value = {
                "detail": [
                    {"msg": "field required", "field": "display_name"},
                    {"msg": "invalid email", "field": "email"},
                ]
            }
            mock_response.text = '{"detail": []}'
            mock_req.return_value = mock_response

            with pytest.raises(ValidationError) as exc_info:
                self.client.register_provider("", "", "invalid-email")

            assert exc_info.value.status_code == 422
            assert "field required" in str(exc_info.value)

    def test_error_handling_server_error(self):
        """Test ServerError handling."""
        with patch("modelsignature.client.requests.Session.request") as mock_req:
            mock_response_1 = MagicMock()
            mock_response_1.status_code = 503
            mock_response_1.json.return_value = {
                "detail": "Service temporarily unavailable"
            }
            mock_response_1.text = '{"detail": "Service temporarily unavailable"}'

            mock_response_2 = MagicMock()
            mock_response_2.status_code = 503
            mock_response_2.json.return_value = {
                "detail": "Service temporarily unavailable"
            }
            mock_response_2.text = '{"detail": "Service temporarily unavailable"}'

            mock_response_3 = MagicMock()
            mock_response_3.status_code = 500
            mock_response_3.json.return_value = {"detail": "Internal server error"}
            mock_response_3.text = '{"detail": "Internal server error"}'

            # Mock multiple attempts due to retries
            mock_req.side_effect = [
                mock_response_1,
                mock_response_2,
                mock_response_3,
            ]

            with pytest.raises(ServerError) as exc_info:
                self.client.verify_token("test_token")

            assert exc_info.value.status_code == 500
            assert "Internal server error" in str(exc_info.value)

    def test_enums_usage(self):
        """Test that enums work correctly."""
        # Test ModelCapability enum
        assert ModelCapability.TEXT_GENERATION == "text-generation"
        assert ModelCapability.REASONING == "reasoning"

        # Test InputType enum
        assert InputType.TEXT == "text"
        assert InputType.IMAGE == "image"

        # Test OutputType enum
        assert OutputType.TEXT == "text"
        assert OutputType.JSON == "json"

        # Test IncidentCategory enum
        assert IncidentCategory.HARMFUL_CONTENT == "harmful_content"
        assert IncidentCategory.TECHNICAL_ERROR == "technical_error"

        # Test IncidentSeverity enum
        assert IncidentSeverity.LOW == "low"
        assert IncidentSeverity.HIGH == "high"

    @patch("modelsignature.client.ModelSignatureClient._request")
    def test_incident_reporting_with_enums(self, mock_request):
        """Test incident reporting using enum values."""
        mock_request.return_value = {"status": "success", "incident_id": "inc_123"}

        response = self.client.report_incident(
            model_id="model_123",
            category=IncidentCategory.HARMFUL_CONTENT.value,
            title="Model generated harmful content",
            description="The model produced inappropriate content",
            severity=IncidentSeverity.HIGH.value,
        )

        assert response["status"] == "success"
        assert response["incident_id"] == "inc_123"

        # Verify correct enum values were sent
        call_args = mock_request.call_args
        json_data = call_args[1]["json"]
        assert json_data["category"] == "harmful_content"
        assert json_data["severity"] == "high"
