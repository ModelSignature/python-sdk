"""ModelSignature Python SDK."""

__version__ = "0.2.1"

from typing import Any, Dict

from .client import ModelSignatureClient
from .identity import IdentityQuestionDetector
from .exceptions import (
    ModelSignatureError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    NetworkError,
    ConflictError,
    NotFoundError,
    PermissionError,
    ServerError,
)
from .models import (
    ModelCapability,
    InputType,
    OutputType,
    TrustLevel,
    IncidentCategory,
    IncidentSeverity,
    HeadquartersLocation,
    VerificationResponse,
    ProviderResponse,
    ModelResponse,
    ApiKeyResponse,
    ApiKeyCreateResponse,
)

# Embedding functionality (optional dependencies)
try:
    from .embedding import embed_signature_link

    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_AVAILABLE = False

    def embed_signature_link(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise ImportError(
            "Embedding functionality requires additional dependencies. "
            "Install with: pip install 'modelsignature[embedding]'"
        )


__all__ = [
    "ModelSignatureClient",
    "IdentityQuestionDetector",
    "ModelSignatureError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
    "NetworkError",
    "ConflictError",
    "NotFoundError",
    "PermissionError",
    "ServerError",
    "ModelCapability",
    "InputType",
    "OutputType",
    "TrustLevel",
    "IncidentCategory",
    "IncidentSeverity",
    "HeadquartersLocation",
    "VerificationResponse",
    "ProviderResponse",
    "ModelResponse",
    "ApiKeyResponse",
    "ApiKeyCreateResponse",
    "embed_signature_link",
]
