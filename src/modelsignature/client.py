from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import time
import uuid
import random
import re
from datetime import datetime
import requests  # type: ignore[import]
from urllib.parse import urljoin

from .exceptions import (
    ModelSignatureError,
    AuthenticationError,
    ValidationError,
    NetworkError,
    RateLimitError,
    ConflictError,
    NotFoundError,
    PermissionError,
    ServerError,
)
from .models import (
    VerificationResponse,
    ModelResponse,
    ProviderResponse,
    HeadquartersLocation,
    ApiKeyResponse,
    ApiKeyCreateResponse,
)
from .constants import DEFAULT_BASE_URL, DEFAULT_TIMEOUT


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime string, handling 'Z' suffix for UTC."""
    if not dt_str:
        return None
    # Replace 'Z' with '+00:00' for Python 3.9 compatibility
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


class ModelSignatureClient:
    """ModelSignature API client for Python."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        debug: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "modelsignature-python/0.2.0"
        self._verification_cache: Dict[tuple, VerificationResponse] = {}
        if api_key:
            self._session.headers["X-API-Key"] = api_key
        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    def create_verification(
        self,
        model_id: str,
        user_fingerprint: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResponse:
        """Create a verification token for your model."""
        if not model_id or not re.match(r"^[A-Za-z0-9_-]+$", model_id):
            raise ValidationError("Invalid model_id format")
        if not user_fingerprint:
            raise ValidationError("user_fingerprint cannot be empty")

        cache_key = (model_id, user_fingerprint)
        cached = self._verification_cache.get(cache_key)
        if cached and not cached.is_expired:
            return cached

        data: Dict[str, Any] = {
            "model_id": model_id,
            "user_fingerprint": user_fingerprint,
        }
        if metadata:
            data["metadata"] = metadata

        resp = self._request("POST", "/api/v1/create-verification", json=data)
        verification = VerificationResponse(
            verification_url=resp["verification_url"],
            token=resp["token"],
            expires_in=resp["expires_in"],
            raw_response=resp,
        )
        self._verification_cache[cache_key] = verification
        return verification

    def verify_token(self, token: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/verify/{token}")

    def register_provider(
        self, company_name: str, email: str, website: str, **kwargs
    ) -> ProviderResponse:
        data = {
            "company_name": company_name,
            "email": email,
            "website": website,
        }
        data.update(kwargs)
        resp = self._request("POST", "/api/v1/providers/register", json=data)
        return ProviderResponse(
            provider_id=str(resp.get("provider_id", "")),
            api_key=str(resp.get("api_key", "")),
            message=resp.get("message", ""),
            trust_center_url=resp.get("trust_center_url"),
            github_url=resp.get("github_url"),
            linkedin_url=resp.get("linkedin_url"),
            raw_response=resp,
        )

    def update_provider(
        self,
        provider_id: str,
        company_name: Optional[str] = None,
        email: Optional[str] = None,
        website: Optional[str] = None,
        trust_center_url: Optional[str] = None,
        github_url: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Update provider details."""

        data: Dict[str, Any] = {}
        if company_name is not None:
            data["company_name"] = company_name
        if email is not None:
            data["email"] = email
        if website is not None:
            data["website"] = website
        if trust_center_url is not None:
            data["trust_center_url"] = trust_center_url
        if github_url is not None:
            data["github_url"] = github_url
        if linkedin_url is not None:
            data["linkedin_url"] = linkedin_url
        data.update(kwargs)

        resp = self._request(
            "PATCH",
            f"/api/v1/providers/{provider_id}",
            json=data,
        )
        return ProviderResponse(
            provider_id=str(resp.get("provider_id", provider_id)),
            api_key=str(resp.get("api_key", "")),
            message=resp.get("message", ""),
            trust_center_url=resp.get("trust_center_url"),
            github_url=resp.get("github_url"),
            linkedin_url=resp.get("linkedin_url"),
            raw_response=resp,
        )

    def update_provider_profile(
        self,
        provider_id: str,
        company_name: Optional[str] = None,
        website: Optional[str] = None,
        description: Optional[str] = None,
        founded_year: Optional[int] = None,
        headquarters_location: Optional[HeadquartersLocation] = None,
        employee_count: Optional[str] = None,
        phone_number: Optional[str] = None,
        support_email: Optional[str] = None,
        logo_url: Optional[str] = None,
        trust_center_url: Optional[str] = None,
        github_url: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update provider profile with complete information."""

        data: Dict[str, Any] = {}
        if company_name is not None:
            data["company_name"] = company_name
        if website is not None:
            data["website"] = website
        if description is not None:
            data["description"] = description
        if founded_year is not None:
            data["founded_year"] = founded_year
        if headquarters_location is not None:
            data["headquarters_location"] = {
                "city": headquarters_location.city,
                "state": headquarters_location.state,
                "country": headquarters_location.country,
            }
        if employee_count is not None:
            data["employee_count"] = employee_count
        if phone_number is not None:
            data["phone_number"] = phone_number
        if support_email is not None:
            data["support_email"] = support_email
        if logo_url is not None:
            data["logo_url"] = logo_url
        if trust_center_url is not None:
            data["trust_center_url"] = trust_center_url
        if github_url is not None:
            data["github_url"] = github_url
        if linkedin_url is not None:
            data["linkedin_url"] = linkedin_url
        data.update(kwargs)

        return self._request(
            "PUT",
            f"/api/v1/providers/{provider_id}/profile",
            json=data,
        )

    def update_provider_compliance(
        self,
        provider_id: str,
        compliance_certifications: Optional[List[str]] = None,
        ai_specific_certifications: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update provider compliance information."""

        data: Dict[str, Any] = {}
        if compliance_certifications is not None:
            data["compliance_certifications"] = compliance_certifications
        if ai_specific_certifications is not None:
            data["ai_specific_certifications"] = ai_specific_certifications
        data.update(kwargs)

        return self._request(
            "PUT",
            f"/api/v1/providers/{provider_id}/compliance",
            json=data,
        )

    def register_model(
        self,
        display_name: str,
        api_model_identifier: str,
        endpoint: str,
        version: str,
        description: str,
        model_type: str,
        family_name: Optional[str] = None,
        model_family_id: Optional[str] = None,
        is_public: bool = True,
        force_new_version: bool = False,
        release_date: Optional[str] = None,
        training_cutoff: Optional[str] = None,
        architecture: Optional[str] = None,
        context_window: Optional[int] = None,
        model_size_params: Optional[str] = None,
        model_card_url: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        input_types: Optional[List[str]] = None,
        output_types: Optional[List[str]] = None,
        serving_regions: Optional[List[str]] = None,
        uptime_sla: Optional[float] = None,
        avg_response_time: Optional[int] = None,
        performance_benchmarks: Optional[Dict[str, Any]] = None,
        huggingface_model_id: Optional[str] = None,
        enable_health_monitoring: bool = False,
        security_standards: Optional[Dict[str, Any]] = None,
        github_repo_url: Optional[str] = None,
        huggingface_url: Optional[str] = None,
        paper_url: Optional[str] = None,
        tls_version: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """Register a model with metadata for verification."""

        data = {
            "display_name": display_name,
            "api_model_identifier": api_model_identifier,
            "endpoint": endpoint,
            "version": version,
            "description": description,
            "model_type": model_type,
            "family_name": family_name,
            "model_family_id": model_family_id,
            "is_public": is_public,
            "force_new_version": force_new_version,
            "release_date": release_date,
            "training_cutoff": training_cutoff,
            "architecture": architecture,
            "context_window": context_window,
            "model_size_params": model_size_params,
            "model_card_url": model_card_url,
            "capabilities": capabilities,
            "input_types": input_types,
            "output_types": output_types,
            "serving_regions": serving_regions,
            "uptime_sla": uptime_sla,
            "avg_response_time": avg_response_time,
            "performance_benchmarks": performance_benchmarks,
            "huggingface_model_id": huggingface_model_id,
            "enable_health_monitoring": enable_health_monitoring,
            "security_standards": security_standards,
            "github_repo_url": github_repo_url,
            "huggingface_url": huggingface_url,
            "paper_url": paper_url,
            "tls_version": tls_version,
        }
        # Remove None values so we don't send them to the API
        data = {k: v for k, v in data.items() if v is not None}
        data.update(kwargs)

        resp = self._request("POST", "/api/v1/models/register", json=data)
        return ModelResponse(
            model_id=str(resp.get("model_id", "")),
            name=resp.get("display_name", display_name),
            version=resp.get("version", version),
            version_number=resp.get("version_number"),
            message=resp.get("message", ""),
            can_quick_edit=resp.get("can_quick_edit"),
            raw_response=resp,
        )

    def sync_huggingface_model(self, model_id: str) -> Dict[str, Any]:
        """Sync model information from HuggingFace"""
        return self._request(
            "POST",
            f"/api/v1/models/{model_id}/sync-huggingface",
        )

    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """Get model health status"""
        return self._request("GET", f"/api/v1/models/{model_id}/health")

    def report_incident(
        self,
        model_id: str,
        category: str,
        title: str,
        description: str,
        verification_token: Optional[str] = None,
        severity: str = "medium",
        reporter_email: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Report an incident for a model."""

        data: Dict[str, Any] = {
            "model_id": model_id,
            "category": category,
            "title": title,
            "description": description,
            "severity": severity,
        }

        if verification_token:
            data["verification_token"] = verification_token
        if reporter_email:
            data["reporter_email"] = reporter_email
        data.update(kwargs)

        return self._request("POST", "/api/v1/incidents/report", json=data)

    def get_my_incidents(
        self,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get incidents reported for your models (provider only)."""

        params = {"status": status} if status else {}
        resp = self._request(
            "GET",
            "/api/v1/providers/me/incidents",
            params=params,
        )
        return resp.get("incidents", [])

    def report_harmful_content(
        self,
        model_id: str,
        content_description: str,
        verification_token: Optional[str] = None,
        severity: str = "high",
    ) -> Dict[str, Any]:
        """Convenience method for reporting harmful content generation."""

        return self.report_incident(
            model_id=model_id,
            category="harmful_content",
            title="Generated harmful content",
            description=content_description,
            verification_token=verification_token,
            severity=severity,
        )

    def report_technical_error(
        self,
        model_id: str,
        error_details: str,
        verification_token: Optional[str] = None,
        severity: str = "medium",
    ) -> Dict[str, Any]:
        """Convenience method for reporting technical errors."""

        return self.report_incident(
            model_id=model_id,
            category="technical_error",
            title="Technical error encountered",
            description=error_details,
            verification_token=verification_token,
            severity=severity,
        )

    def report_impersonation(
        self,
        model_id: str,
        impersonation_details: str,
        verification_token: Optional[str] = None,
        severity: str = "high",
    ) -> Dict[str, Any]:
        """Convenience method for reporting model impersonation."""

        return self.report_incident(
            model_id=model_id,
            category="impersonation",
            title="Model impersonation detected",
            description=impersonation_details,
            verification_token=verification_token,
            severity=severity,
        )

    def archive_model(
        self,
        model_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Archive a model and all its versions."""

        data = {}
        if reason is not None:
            data["reason"] = reason

        return self._request(
            "PUT",
            f"/api/v1/models/{model_id}/archive",
            json=data,
        )

    def unarchive_model(self, model_id: str) -> Dict[str, Any]:
        """Unarchive a model."""

        return self._request(
            "PUT",
            f"/api/v1/models/{model_id}/unarchive",
        )

    def update_model_visibility(
        self,
        model_id: str,
        is_public: bool,
    ) -> Dict[str, Any]:
        """Update model visibility settings."""

        data = {"is_public": is_public}
        return self._request(
            "PUT",
            f"/api/v1/models/{model_id}/visibility",
            json=data,
        )

    # ====== DEPLOYMENT MANAGEMENT METHODS (Milestone 1) ======

    def register_deployment(
        self,
        spki_fingerprint: str,
        spiffe_id: Optional[str] = None,
        cloud_identity: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Register a deployment with SPKI fingerprint for mTLS identity.
        
        Args:
            spki_fingerprint: 64-character hex SHA256 hash of client certificate public key
            spiffe_id: Optional SPIFFE identity for the deployment  
            cloud_identity: Optional cloud service account or identity
            labels: Optional metadata labels for the deployment
            
        Returns:
            Dict containing deployment_id and other deployment info
            
        Raises:
            ValidationError: If SPKI fingerprint format is invalid
            ConflictError: If SPKI fingerprint already registered
            AuthenticationError: If API key is invalid
        """
        if not spki_fingerprint or len(spki_fingerprint) != 64:
            raise ValidationError("SPKI fingerprint must be exactly 64 hex characters")
        
        # Validate hex format
        try:
            int(spki_fingerprint, 16)
        except ValueError:
            raise ValidationError("SPKI fingerprint must be valid hex")

        data = {
            "spki_fingerprint": spki_fingerprint.lower(),
        }
        if spiffe_id:
            data["spiffe_id"] = spiffe_id
        if cloud_identity:
            data["cloud_identity"] = cloud_identity
        if labels:
            data["labels"] = labels

        return self._request("POST", "/api/v1/deployments/register", json=data)

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments for the authenticated provider.
        
        Returns:
            List of deployment dictionaries with deployment info
        """
        return self._request("GET", "/api/v1/deployments")

    def allow_deployment_for_model(
        self,
        model_id: str,
        deployment_id: str,
    ) -> Dict[str, Any]:
        """Allow a deployment to create verifications for a model.
        
        Args:
            model_id: ID of the model to authorize
            deployment_id: ID of the deployment to authorize
            
        Returns:
            Dict with success confirmation
            
        Raises:
            NotFoundError: If model or deployment not found
            ConflictError: If deployment already authorized for model
            PermissionError: If deployment doesn't belong to provider
        """
        data = {"deployment_id": deployment_id}
        return self._request("POST", f"/api/v1/models/{model_id}/allow-deployment", json=data)

    def update_deployment_status(
        self,
        deployment_id: str,
        status: str,
    ) -> Dict[str, Any]:
        """Enable or disable a deployment.
        
        Args:
            deployment_id: ID of the deployment to update
            status: New status ("enabled" or "disabled")
            
        Returns:
            Dict with success confirmation
            
        Raises:
            ValidationError: If status is not "enabled" or "disabled"
            NotFoundError: If deployment not found
        """
        if status not in ["enabled", "disabled"]:
            raise ValidationError('Status must be either "enabled" or "disabled"')
            
        data = {"status": status}
        return self._request("PUT", f"/api/v1/deployments/{deployment_id}/status", json=data)

    def delete_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Delete a deployment and remove all model authorizations.
        
        Args:
            deployment_id: ID of the deployment to delete
            
        Returns:
            Dict with success confirmation
            
        Raises:
            NotFoundError: If deployment not found
        """
        return self._request("DELETE", f"/api/v1/deployments/{deployment_id}")

    def create_verification_with_mtls(
        self,
        model_id: str,
        user_fingerprint: str,
        client_cert_spki: str,
        client_cert_subject: Optional[str] = None,
        client_cert_serial: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResponse:
        """Create a verification token with mTLS client certificate validation.
        
        Args:
            model_id: ID of the model to create verification for
            user_fingerprint: Unique identifier for the user/session
            client_cert_spki: SPKI fingerprint of the client certificate
            client_cert_subject: Optional certificate subject DN
            client_cert_serial: Optional certificate serial number
            metadata: Optional additional metadata
            
        Returns:
            VerificationResponse with token and verification URL
            
        Raises:
            ValidationError: If parameters are invalid
            AuthenticationError: If deployment not authorized for model
            NotFoundError: If model not found
        """
        if not model_id or not re.match(r"^[A-Za-z0-9_-]+$", model_id):
            raise ValidationError("Invalid model_id format")
        if not user_fingerprint:
            raise ValidationError("user_fingerprint cannot be empty")
        if not client_cert_spki or len(client_cert_spki) != 64:
            raise ValidationError("client_cert_spki must be exactly 64 hex characters")

        # Add mTLS headers
        headers = {
            "x-ms-client-spki": client_cert_spki.lower(),
        }
        if client_cert_subject:
            headers["x-ms-client-subject"] = client_cert_subject
        if client_cert_serial:
            headers["x-ms-client-serial"] = client_cert_serial

        data: Dict[str, Any] = {
            "model_id": model_id,
            "user_fingerprint": user_fingerprint,
        }
        if metadata:
            data["metadata"] = metadata

        resp = self._request("POST", "/api/v1/create-verification", json=data, headers=headers)
        verification = VerificationResponse(
            verification_url=resp["verification_url"],
            token=resp["token"],
            expires_in=resp["expires_in"],
            raw_response=resp,
        )

        # Cache the verification
        cache_key = (model_id, user_fingerprint)
        self._verification_cache[cache_key] = verification

        return verification

    def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """Get version history for a model."""

        return self._request(
            "GET",
            f"/api/v1/models/{model_id}/history",
        )

    def get_latest_model_version(self, model_id: str) -> Dict[str, Any]:
        """Get the latest version of a model by any version's ID."""

        return self._request(
            "GET",
            f"/api/v1/models/{model_id}/latest",
        )

    def get_model_community_stats(self, model_id: str) -> Dict[str, Any]:
        """Get community statistics for a model."""

        return self._request(
            "GET",
            f"/api/v1/models/{model_id}/community-stats",
        )

    def list_api_keys(self) -> List[ApiKeyResponse]:
        """List all API keys for the authenticated provider."""

        resp = self._request("GET", "/api/v1/providers/me/api-keys")
        keys_data = resp if isinstance(resp, list) else resp.get("keys", [])
        return [
            ApiKeyResponse(
                id=key["id"],
                name=key["name"],
                key_prefix=key["key_prefix"],
                last_used_at=_parse_datetime(key.get("last_used_at")),
                is_active=key["is_active"],
                created_at=_parse_datetime(key.get("created_at")),
            )
            for key in keys_data
        ]

    def create_api_key(self, name: str) -> ApiKeyCreateResponse:
        """Create a new API key for the authenticated provider."""

        data = {"name": name}
        resp = self._request(
            "POST", "/api/v1/providers/me/api-keys", json=data
        )

        created_at = _parse_datetime(resp["created_at"])
        if created_at is None:
            raise ValueError("Invalid created_at timestamp")
        return ApiKeyCreateResponse(
            id=resp["id"],
            name=resp["name"],
            key_prefix=resp["key_prefix"],
            api_key=resp["api_key"],
            created_at=created_at,
        )

    def revoke_api_key(self, key_id: str) -> Dict[str, Any]:
        """Revoke (deactivate) an API key."""

        return self._request(
            "DELETE",
            f"/api/v1/providers/me/api-keys/{key_id}",
        )

    def search(
        self,
        query: str,
        limit: int = 20,
        include_providers: bool = True,
        include_models: bool = True,
    ) -> Dict[str, Any]:
        """Search across providers and models."""

        params = {
            "q": query,
            "limit": limit,
            "include_providers": include_providers,
            "include_models": include_models,
        }

        return self._request("GET", "/api/v1/search", params=params)

    def list_public_models(
        self,
        skip: int = 0,
        limit: int = 1000,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all public models."""

        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if provider_id:
            params["provider_id"] = provider_id

        resp = self._request("GET", "/api/v1/models/public", params=params)
        return resp if isinstance(resp, list) else resp.get("models", [])

    def list_public_providers(
        self,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List all public providers."""

        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        resp = self._request("GET", "/api/v1/providers/public", params=params)
        return resp if isinstance(resp, list) else resp.get("providers", [])

    def get_public_model(self, model_id: str) -> Dict[str, Any]:
        """Get public model information."""

        return self._request("GET", f"/api/v1/models/{model_id}/public")

    def get_public_provider(self, provider_id: str) -> Dict[str, Any]:
        """Get public provider information."""

        return self._request("GET", f"/api/v1/providers/{provider_id}/public")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        backoff = [1, 2, 4]
        for attempt in range(self.max_retries):
            req_id = str(uuid.uuid4())
            headers = dict(kwargs.get("headers", {}))
            headers.setdefault("User-Agent", "modelsignature-python/0.2.0")
            headers["X-Request-ID"] = req_id

            start = time.time()
            try:
                resp = self._session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    headers=headers,
                    **kwargs,
                )
            except requests.RequestException as exc:
                logging.debug("Request %s failed: %s", req_id, exc)
                if attempt >= self.max_retries - 1:
                    raise NetworkError(str(exc))
                delay = float(backoff[min(attempt, len(backoff) - 1)])
                delay *= 0.5 + random.random()
                time.sleep(delay)
                continue

            duration = int((time.time() - start) * 1000)
            logging.debug(
                "[%s] %s %s -> %s (%dms)",
                req_id,
                method,
                endpoint,
                resp.status_code,
                duration,
            )
            if duration > 1000:
                logging.warning("Slow request %s took %dms", req_id, duration)

            if resp.status_code == 401:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                raise AuthenticationError(
                    detail,
                    status_code=401,
                    response=resp.json() if resp.text else {},
                )
            if resp.status_code == 403:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                raise PermissionError(
                    detail,
                    status_code=403,
                    response=resp.json() if resp.text else {},
                )
            if resp.status_code == 404:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                raise NotFoundError(
                    detail,
                    status_code=404,
                    response=resp.json() if resp.text else {},
                )
            if resp.status_code == 409:
                try:
                    resp_json = resp.json()
                    detail = resp_json.get("message", resp.text)
                    existing = resp_json.get("existing_model")
                except ValueError:
                    detail = resp.text
                    existing = None
                raise ConflictError(
                    detail,
                    existing_resource=existing,
                    status_code=409,
                    response=resp_json if resp.text else {},
                )
            if resp.status_code == 422:
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict):
                        # fmt: off
                        errors = (
                            err_json.get("errors")
                            or err_json.get("detail")
                        )
                        # fmt: on
                        if isinstance(errors, list):
                            # fmt: off
                            detail = "; ".join(
                                e.get("msg", str(e)) for e in errors
                            )
                            # fmt: on
                        else:
                            detail = str(errors)
                    else:
                        detail = str(err_json)
                except ValueError:
                    detail = resp.text
                    err_json = {}
                raise ValidationError(
                    f"Invalid parameters: {detail}",
                    errors=err_json,
                    status_code=422,
                    response=err_json,
                )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                if attempt >= self.max_retries - 1:
                    raise RateLimitError(
                        "Rate limit exceeded. Retry after {n} seconds".format(
                            n=retry_after
                        ),
                        retry_after,
                    )
                time.sleep(retry_after)
                continue

            if resp.status_code in {502, 503, 504}:
                if attempt >= self.max_retries - 1:
                    # fmt: off
                    raise NetworkError(
                        "ModelSignature API is temporarily unavailable"
                    )
                    # fmt: on
                delay = float(backoff[min(attempt, len(backoff) - 1)])
                delay *= 0.5 + random.random()
                time.sleep(delay)
                continue

            if resp.status_code >= 500:
                try:
                    detail = resp.json().get("detail", resp.text)
                    resp_json = resp.json()
                except ValueError:
                    detail = resp.text
                    resp_json = {}
                if attempt >= self.max_retries - 1:
                    # fmt: off
                    raise ServerError(
                        f"Server error {resp.status_code}: {detail}",
                        status_code=resp.status_code,
                        response=resp_json
                    )
                    # fmt: on
                delay = float(backoff[min(attempt, len(backoff) - 1)])
                delay *= 0.5 + random.random()
                time.sleep(delay)
                continue

            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    raise ModelSignatureError("Invalid JSON response")

            if attempt >= self.max_retries - 1:
                raise ModelSignatureError(
                    f"API Error {resp.status_code}: {resp.text}"  # noqa: E501
                )
            time.sleep(backoff[min(attempt, len(backoff) - 1)])

        raise ModelSignatureError("Request failed")
