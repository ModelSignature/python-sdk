from typing import Optional


class ModelSignatureError(Exception):
    """Base exception for ModelSignature SDK."""

    pass


class AuthenticationError(ModelSignatureError):
    """Raised when API key is invalid or missing."""

    pass


class ValidationError(ModelSignatureError):
    """Raised when request parameters are invalid."""

    pass


class RateLimitError(ModelSignatureError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(ModelSignatureError):
    """Raised when network request fails."""

    pass
