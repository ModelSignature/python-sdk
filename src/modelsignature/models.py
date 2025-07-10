from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime, timedelta


@dataclass
class VerificationResponse:
    """Response from create_verification endpoint."""

    verification_url: str
    token: str
    expires_in: int
    raw_response: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        # Assume raw_response has 'created_at' timestamp in ISO format
        created = self.raw_response.get("created_at")
        if not created:
            return False
        created_dt = datetime.fromisoformat(created)
        expiry = created_dt + timedelta(seconds=self.expires_in)
        return datetime.utcnow() > expiry


@dataclass
class ProviderResponse:
    """Response from provider registration."""

    provider_id: str
    api_key: str
    message: str
    raw_response: Dict[str, Any]


@dataclass
class ModelResponse:
    """Response from model registration."""

    model_id: str
    name: str
    version: str
    message: str
    raw_response: Dict[str, Any]
