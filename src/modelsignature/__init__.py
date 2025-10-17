"""ModelSignature Python SDK."""

__version__ = "0.2.1"

from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
if TYPE_CHECKING:
    # Always import for type checking
    from .embedding import embed_signature_link
else:
    try:
        from .embedding import embed_signature_link

        _EMBEDDING_AVAILABLE = True
    except ImportError:
        _EMBEDDING_AVAILABLE = False

        def embed_signature_link(  # type: ignore[misc]
            model: str,
            link: str,
            api_key: Optional[str] = None,
            out_dir: Optional[str] = None,
            mode: str = "adapter",
            fp: str = "4bit",
            rank: int = 32,
            alpha: Optional[int] = None,
            dropout: float = 0.1,
            epochs: int = 10,
            learning_rate: float = 2e-4,
            batch_size: int = 1,
            gradient_accumulation_steps: int = 8,
            dataset_size: int = 500,
            custom_triggers: Optional[List[str]] = None,
            custom_responses: Optional[List[str]] = None,
            push_to_hf: bool = False,
            hf_repo_id: Optional[str] = None,
            hf_token: Optional[str] = None,
            evaluate: bool = True,
            debug: bool = False,
            **kwargs: Any,
        ) -> Dict[str, Any]:
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
