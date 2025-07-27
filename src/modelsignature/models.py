from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum


class ModelCapability(str, Enum):
    """Model capabilities enum matching API schema."""
    TEXT_GENERATION = "text-generation"
    REASONING = "reasoning"
    CODE_GENERATION = "code-generation"
    MATH = "math"
    CREATIVE_WRITING = "creative-writing"
    CONVERSATION = "conversation"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    IMAGE_GENERATION = "image-generation"
    IMAGE_UNDERSTANDING = "image-understanding"
    AUDIO_PROCESSING = "audio-processing"
    VIDEO_PROCESSING = "video-processing"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function-calling"


class InputType(str, Enum):
    """Input types enum matching API schema."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"
    CODE = "code"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"


class OutputType(str, Enum):
    """Output types enum matching API schema."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"
    CODE = "code"
    STRUCTURED_DATA = "structured-data"
    MARKDOWN = "markdown"
    HTML = "html"


class TrustLevel(str, Enum):
    """Trust levels enum matching API schema."""
    UNVERIFIED = "unverified"
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    PREMIUM = "premium"


class IncidentCategory(str, Enum):
    """Incident categories enum matching API schema."""
    HARMFUL_CONTENT = "harmful_content"
    TECHNICAL_ERROR = "technical_error"
    IMPERSONATION = "impersonation"
    PRIVACY_VIOLATION = "privacy_violation"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    """Incident severity enum matching API schema."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HeadquartersLocation:
    """Headquarters location data."""
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


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
    trust_center_url: Optional[str] = None
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None


@dataclass
class ModelResponse:
    """Response from model registration."""

    model_id: str
    name: str
    version: str
    message: str
    raw_response: Dict[str, Any]
    version_number: Optional[int] = None
    can_quick_edit: Optional[bool] = None


@dataclass
class ApiKeyResponse:
    """Response from API key operations."""
    
    id: str
    name: str
    key_prefix: str
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    created_at: Optional[datetime] = None


@dataclass 
class ApiKeyCreateResponse:
    """Response from creating a new API key."""
    
    id: str
    name: str
    key_prefix: str
    api_key: str  # Full API key - only returned on creation
    created_at: datetime
