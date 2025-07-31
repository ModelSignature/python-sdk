from typing import Optional, Dict, Any, List


class ModelSignatureError(Exception):
    """Base exception for ModelSignature SDK."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class AuthenticationError(ModelSignatureError):
    """Raised when API key is invalid or missing."""

    pass


class ValidationError(ModelSignatureError):
    """Raised when request parameters are invalid."""

    def __init__(
        self, message: str, errors: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.errors = errors or {}


class RateLimitError(ModelSignatureError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self, message: str, retry_after: Optional[int] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class NetworkError(ModelSignatureError):
    """Raised when network request fails."""

    pass


class ConflictError(ModelSignatureError):
    """Raised when a resource conflict occurs (409 status)."""

    def __init__(
        self,
        message: str,
        existing_resource: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.existing_resource = existing_resource or {}


class NotFoundError(ModelSignatureError):
    """Raised when a requested resource is not found (404 status)."""

    pass


class PermissionError(ModelSignatureError):
    """Raised when user lacks permission for requested action (403 status)."""

    pass


class ServerError(ModelSignatureError):
    """Raised when the server encounters an internal error (5xx status)."""

    pass


class PolicyViolationError(ModelSignatureError):
    """Raised when a policy is violated."""
    
    def __init__(self, message: str, violations: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.violations = violations or []
