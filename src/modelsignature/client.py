from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging
import time
import uuid
import random
import re
import requests  # type: ignore[import]
from urllib.parse import urljoin

from .exceptions import (
    ModelSignatureError,
    AuthenticationError,
    ValidationError,
    NetworkError,
    RateLimitError,
)
from .models import VerificationResponse, ModelResponse, ProviderResponse
from .constants import DEFAULT_BASE_URL, DEFAULT_TIMEOUT


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
        self._session.headers["User-Agent"] = "modelsignature-python/0.1.0"
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
            pythontrust_center_url=resp.get("pythontrust_center_url"),
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
        pythontrust_center_url: Optional[str] = None,
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
        if pythontrust_center_url is not None:
            data["pythontrust_center_url"] = pythontrust_center_url
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
            pythontrust_center_url=resp.get("pythontrust_center_url"),
            github_url=resp.get("github_url"),
            linkedin_url=resp.get("linkedin_url"),
            raw_response=resp,
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
        is_public: bool = True,
        force_new_version: bool = False,
        release_date: Optional[str] = None,
        training_cutoff: Optional[str] = None,
        architecture: Optional[str] = None,
        context_window: Optional[int] = None,
        model_size_params: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        huggingface_model_id: Optional[str] = None,
        enable_health_monitoring: bool = False,
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
            "is_public": is_public,
            "force_new_version": force_new_version,
            "release_date": release_date,
            "training_cutoff": training_cutoff,
            "architecture": architecture,
            "context_window": context_window,
            "model_size_params": model_size_params,
            "capabilities": capabilities,
            "huggingface_model_id": huggingface_model_id,
            "enable_health_monitoring": enable_health_monitoring,
        }
        # Remove None values so we don't send them to the API
        data = {k: v for k, v in data.items() if v is not None}
        data.update(kwargs)

        resp = self._request("POST", "/api/v1/models/register", json=data)
        return ModelResponse(
            model_id=str(resp.get("model_id", "")),
            name=resp.get("name", display_name),
            version=resp.get("version", version),
            message=resp.get("message", ""),
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

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        backoff = [1, 2, 4]
        for attempt in range(self.max_retries):
            req_id = str(uuid.uuid4())
            headers = dict(kwargs.get("headers", {}))
            headers.setdefault("User-Agent", "modelsignature-python/0.1.0")
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

            if resp.status_code in {401, 403}:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                raise AuthenticationError(detail)
            if resp.status_code == 404:
                raise ValidationError(
                    "Model ID not found. Register at modelsignature.com"
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
                raise ValidationError(f"Invalid parameters: {detail}")

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
                except ValueError:
                    detail = resp.text
                if attempt >= self.max_retries - 1:
                    # fmt: off
                    raise NetworkError(
                        f"Server error {resp.status_code}: {detail}"
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
