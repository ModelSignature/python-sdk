from __future__ import annotations

from typing import Optional, Dict, Any
import requests
from urllib.parse import urljoin

from .exceptions import (
    ModelSignatureError,
    AuthenticationError,
    ValidationError,
    NetworkError,
    RateLimitError,
)
from .models import VerificationResponse, ModelResponse, ProviderResponse
from .auth import AuthHandler
from .constants import DEFAULT_BASE_URL, DEFAULT_TIMEOUT


class ModelSignatureClient:
    """ModelSignature API client for Python."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def create_verification(
        self, model_id: str, user_fingerprint: str, **kwargs
    ) -> VerificationResponse:
        data = {"model_id": model_id, "user_fingerprint": user_fingerprint}
        data.update(kwargs)
        resp = self._request("POST", "/api/v1/create-verification", json=data)
        return VerificationResponse(
            verification_url=resp["verification_url"],
            token=resp["token"],
            expires_in=resp["expires_in"],
            raw_response=resp,
        )

    def verify_token(self, token: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/verify/{token}")

    def register_provider(
        self, company_name: str, email: str, website: str, **kwargs
    ) -> ProviderResponse:
        data = {"company_name": company_name, "email": email, "website": website}
        data.update(kwargs)
        resp = self._request("POST", "/api/v1/providers/register", json=data)
        return ProviderResponse(
            provider_id=resp.get("provider_id"),
            api_key=resp.get("api_key"),
            message=resp.get("message", ""),
            raw_response=resp,
        )

    def register_model(
        self,
        model_name: str,
        version: str,
        description: str,
        api_endpoint: str,
        model_type: str,
        **kwargs,
    ) -> ModelResponse:
        data = {
            "model_name": model_name,
            "version": version,
            "description": description,
            "api_endpoint": api_endpoint,
            "model_type": model_type,
        }
        data.update(kwargs)
        resp = self._request("POST", "/api/v1/models/register", json=data)
        return ModelResponse(
            model_id=resp.get("model_id"),
            name=resp.get("name", model_name),
            version=resp.get("version", version),
            message=resp.get("message", ""),
            raw_response=resp,
        )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        for attempt in range(self.max_retries):
            try:
                resp = self._session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries - 1:
                    raise NetworkError(str(exc))
                continue
            if resp.status_code == 401:
                raise AuthenticationError("Invalid API key")
            if resp.status_code == 400:
                raise ValidationError(resp.text)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "0"))
                raise RateLimitError("Rate limit exceeded", retry_after)
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    raise ModelSignatureError("Invalid JSON response")
            else:
                if attempt >= self.max_retries - 1:
                    raise ModelSignatureError(
                        f"API Error {resp.status_code}: {resp.text}"
                    )
        raise ModelSignatureError("Request failed")
