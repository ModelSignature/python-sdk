from __future__ import annotations

from typing import Optional, Dict, Any
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
            model_id=str(resp.get("model_id", "")),
            name=resp.get("name", model_name),
            version=resp.get("version", version),
            message=resp.get("message", ""),
            raw_response=resp,
        )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        backoff = [1, 2, 4]
        for attempt in range(self.max_retries):
            req_id = str(uuid.uuid4())
            headers = kwargs.pop("headers", {})
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
                delay = backoff[min(attempt, len(backoff) - 1)]
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
                        errors = (
                            err_json.get("errors")
                            or err_json.get("detail")
                        )
                        if isinstance(errors, list):
                            detail = "; ".join(
                                e.get("msg", str(e)) for e in errors
                            )
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
                    raise NetworkError(
                        "ModelSignature API is temporarily unavailable"
                    )
                delay = backoff[min(attempt, len(backoff) - 1)]
                delay *= 0.5 + random.random()
                time.sleep(delay)
                continue

            if resp.status_code >= 500:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                if attempt >= self.max_retries - 1:
                    raise NetworkError(
                        f"Server error {resp.status_code}: {detail}"
                    )
                delay = backoff[min(attempt, len(backoff) - 1)]
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
