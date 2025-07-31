"""
ModelSignature Python SDK - Policy Enforcement
Milestone 4: Client SDKs + policy enforcement
"""

from __future__ import annotations

import hashlib
import json
import base64
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .client import ModelSignatureClient
from .exceptions import PolicyViolationError, ModelSignatureError


@dataclass
class JWTClaims:
    """JWT token claims."""
    model_id: str
    provider_id: str
    user_fp: str
    deployment_id: Optional[str] = None
    model_digest: Optional[str] = None
    response_hash: Optional[str] = None
    bound_to_response: bool = False
    iat: int = 0
    exp: int = 0
    jti: str = ""


@dataclass
class ModelInfo:
    """Model information from verification."""
    id: str
    name: str
    version: str
    type: Optional[str] = None
    model_digest: Optional[str] = None
    sigstore_bundle_url: Optional[str] = None
    bundle_status: Optional[str] = None
    bundle_last_checked: Optional[str] = None


@dataclass
class ProviderInfo:
    """Provider information from verification."""
    id: str
    name: str
    website: Optional[str] = None
    verification_level: Optional[str] = None
    domain_verified: Optional[bool] = None


@dataclass
class BundleCheck:
    """Bundle verification information."""
    status: str
    last_checked: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class VerificationResult:
    """Result of JWT token verification."""
    valid: bool
    claims: Optional[JWTClaims] = None
    model: Optional[ModelInfo] = None
    provider: Optional[ProviderInfo] = None
    bundle_check: Optional[BundleCheck] = None
    error: Optional[str] = None


@dataclass
class PolicyConfig:
    """Policy enforcement configuration."""
    require_deployment_id: bool = False
    require_model_digest: bool = False
    require_bundle_verification: bool = False
    allowed_providers: List[str] = field(default_factory=list)
    allowed_models: List[str] = field(default_factory=list)
    max_token_age: Optional[int] = 900  # 15 minutes default
    fail_closed: bool = True


@dataclass
class PolicyResult:
    """Result of policy enforcement."""
    allowed: bool
    reasons: List[str]
    verification: VerificationResult


def hash_output(response_text: str) -> str:
    """
    Calculate SHA256 hash of model response for binding.
    
    Args:
        response_text: The model's raw response text
        
    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(response_text.encode('utf-8')).hexdigest()


def parse_jwt(token: str) -> Optional[Dict[str, Any]]:
    """
    Parse JWT token without verification (for extracting claims).
    Note: This is for client-side inspection only, not security validation.
    
    Args:
        token: JWT token string
        
    Returns:
        Parsed payload or None if invalid format
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
            
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded.decode('utf-8'))
    except Exception:
        return None


def is_token_expired(token: str) -> Optional[bool]:
    """
    Check if JWT token is expired (client-side check only).
    
    Args:
        token: JWT token string
        
    Returns:
        True if expired, False if valid, None if can't parse
    """
    payload = parse_jwt(token)
    if not payload or 'exp' not in payload:
        return None
    
    now = int(time.time())
    return payload['exp'] < now


def get_token_age(token: str) -> Optional[int]:
    """
    Extract token age in seconds.
    
    Args:
        token: JWT token string
        
    Returns:
        Age in seconds or None if can't parse
    """
    payload = parse_jwt(token)
    if not payload or 'iat' not in payload:
        return None
    
    now = int(time.time())
    return now - payload['iat']


def is_valid_token_format(token: str) -> bool:
    """
    Validate token format (basic structure check).
    
    Args:
        token: JWT token string
        
    Returns:
        True if format is valid
    """
    if not token or not isinstance(token, str):
        return False
    
    parts = token.split('.')
    if len(parts) != 3:
        return False
    
    # Check that each part looks like base64url
    import re
    base64url_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
    return all(base64url_pattern.match(part) for part in parts)


class PolicyEnforcer:
    """Policy enforcement for JWT tokens."""
    
    def __init__(self, client: ModelSignatureClient, config: Optional[PolicyConfig] = None):
        """
        Initialize policy enforcer.
        
        Args:
            client: ModelSignature client for token verification
            config: Policy configuration
        """
        self.client = client
        self.config = config or PolicyConfig()
    
    def enforce_policy(self, token: str) -> PolicyResult:
        """
        Enforce policy on a JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Policy result with allowed status and reasons
            
        Raises:
            PolicyViolationError: If policy is violated and fail_closed is True
        """
        violations = []
        verification = None
        
        try:
            # First verify the token
            verification = self.verify_token(token)
            
            if not verification.valid:
                violations.append(f"Token verification failed: {verification.error}")
                return self._build_result(False, violations, verification)
            
            # Now enforce policy rules
            self._check_token_policy(token, verification, violations)
            
        except Exception as e:
            error_message = str(e)
            violations.append(f"Verification request failed: {error_message}")
            
            verification = VerificationResult(
                valid=False,
                error=error_message
            )
        
        allowed = len(violations) == 0
        result = self._build_result(allowed, violations, verification)
        
        # Throw error if policy violated and fail-closed mode is enabled
        if not allowed and self.config.fail_closed:
            raise PolicyViolationError(
                f"Policy violation: {', '.join(violations)}",
                violations
            )
        
        return result
    
    def verify_token(self, token: str) -> VerificationResult:
        """
        Verify a JWT token using the ModelSignature API.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Verification result
        """
        if not is_valid_token_format(token):
            return VerificationResult(
                valid=False,
                error="Invalid token format"
            )
        
        try:
            # Use the client to call the JWT verification endpoint
            response = self.client._request(
                'GET',
                f'/api/v1/jwt/verify/{token}'
            )
            
            if not response.get('valid', False):
                return VerificationResult(
                    valid=False,
                    error=response.get('error', 'Unknown verification error')
                )
            
            # Parse the response into structured data
            claims_data = response.get('claims', {})
            claims = JWTClaims(
                model_id=claims_data.get('model_id', ''),
                provider_id=claims_data.get('provider_id', ''),
                user_fp=claims_data.get('user_fp', ''),
                deployment_id=claims_data.get('deployment_id'),
                model_digest=claims_data.get('model_digest'),
                response_hash=claims_data.get('response_hash'),
                bound_to_response=claims_data.get('bound_to_response', False),
                iat=claims_data.get('iat', 0),
                exp=claims_data.get('exp', 0),
                jti=claims_data.get('jti', '')
            )
            
            model_data = response.get('model', {})
            model = ModelInfo(
                id=model_data.get('id', ''),
                name=model_data.get('name', ''),
                version=model_data.get('version', ''),
                type=model_data.get('type'),
                model_digest=model_data.get('model_digest'),
                sigstore_bundle_url=model_data.get('sigstore_bundle_url'),
                bundle_status=model_data.get('bundle_status'),
                bundle_last_checked=model_data.get('bundle_last_checked')
            ) if model_data else None
            
            provider_data = response.get('provider', {})
            provider = ProviderInfo(
                id=provider_data.get('id', ''),
                name=provider_data.get('name', ''),
                website=provider_data.get('website'),
                verification_level=provider_data.get('verification_level'),
                domain_verified=provider_data.get('domain_verified')
            ) if provider_data else None
            
            bundle_data = response.get('bundle_check', {})
            bundle_check = BundleCheck(
                status=bundle_data.get('status', 'unknown'),
                last_checked=bundle_data.get('last_checked'),
                details=bundle_data.get('details')
            ) if bundle_data else None
            
            return VerificationResult(
                valid=True,
                claims=claims,
                model=model,
                provider=provider,
                bundle_check=bundle_check
            )
            
        except Exception as e:
            return VerificationResult(
                valid=False,
                error=f"Verification failed: {str(e)}"
            )
    
    def bind_response(self, token: str, response_text: str) -> Dict[str, Any]:
        """
        Bind a JWT token to a specific response text.
        
        Args:
            token: Original JWT token
            response_text: The model's response text
            
        Returns:
            Response binding information
        """
        if not is_valid_token_format(token):
            raise ModelSignatureError("Invalid token format")
        
        if not response_text or not isinstance(response_text, str):
            raise ModelSignatureError("Response text must be a non-empty string")
        
        return self.client._request(
            'POST',
            f'/api/v1/jwt/{token}/bind-response',
            json={'response_text': response_text}
        )
    
    def _check_token_policy(
        self, 
        token: str, 
        verification: VerificationResult, 
        violations: List[str]
    ) -> None:
        """Check token-level policy rules."""
        claims = verification.claims
        model = verification.model
        provider = verification.provider
        bundle_check = verification.bundle_check
        
        if not claims:
            violations.append("No claims found in token")
            return
        
        # Check token age
        if self.config.max_token_age is not None:
            token_age = get_token_age(token)
            if token_age is None:
                violations.append("Unable to determine token age")
            elif token_age > self.config.max_token_age:
                violations.append(f"Token is too old: {token_age}s > {self.config.max_token_age}s")
        
        # Check if deployment ID is required
        if self.config.require_deployment_id and not claims.deployment_id:
            violations.append("Deployment ID is required but not present in token")
        
        # Check if model digest is required
        if self.config.require_model_digest and not claims.model_digest:
            violations.append("Model digest is required but not present in token")
        
        # Check bundle verification requirement
        if self.config.require_bundle_verification:
            if not bundle_check:
                violations.append("Bundle verification is required but no bundle information available")
            elif bundle_check.status != 'verified':
                violations.append(f"Bundle verification failed: status is '{bundle_check.status}'")
        
        # Check allowed providers
        if self.config.allowed_providers:
            if not provider or provider.id not in self.config.allowed_providers:
                violations.append(f"Provider '{provider.id if provider else 'unknown'}' is not in allowed list")
        
        # Check allowed models
        if self.config.allowed_models:
            if not model or model.id not in self.config.allowed_models:
                violations.append(f"Model '{model.id if model else 'unknown'}' is not in allowed list")
    
    def _build_result(
        self, 
        allowed: bool, 
        violations: List[str], 
        verification: Optional[VerificationResult]
    ) -> PolicyResult:
        """Build policy result."""
        return PolicyResult(
            allowed=allowed,
            reasons=violations,
            verification=verification or VerificationResult(valid=False, error="No verification data")
        )
    
    def update_config(self, config: PolicyConfig) -> None:
        """Update policy configuration."""
        self.config = config
    
    def get_config(self) -> PolicyConfig:
        """Get current policy configuration."""
        return self.config
    
    def quick_check(
        self, 
        token: str,
        *,
        require_deployment: bool = False,
        require_digest: bool = False,
        max_age: Optional[int] = None,
        allowed_providers: Optional[List[str]] = None
    ) -> bool:
        """
        Quick policy check with custom rules.
        
        Args:
            token: JWT token
            require_deployment: Require deployment ID
            require_digest: Require model digest
            max_age: Maximum token age in seconds
            allowed_providers: List of allowed provider IDs
            
        Returns:
            Whether token passes policy
        """
        try:
            temp_config = PolicyConfig(
                require_deployment_id=require_deployment,
                require_model_digest=require_digest,
                max_token_age=max_age,
                allowed_providers=allowed_providers or [],
                fail_closed=False  # Don't throw for quick check
            )
            
            temp_enforcer = PolicyEnforcer(self.client, temp_config)
            result = temp_enforcer.enforce_policy(token)
            return result.allowed
        except Exception:
            return False


def create_secure_policy(
    client: ModelSignatureClient, 
    config: Optional[PolicyConfig] = None
) -> PolicyEnforcer:
    """
    Create a policy enforcer with secure defaults for production use.
    
    Args:
        client: ModelSignature client
        config: Additional policy configuration
        
    Returns:
        PolicyEnforcer configured for production use
    """
    secure_config = PolicyConfig(
        fail_closed=True,
        require_deployment_id=True,
        require_model_digest=True,
        max_token_age=300,  # 5 minutes for production
        require_bundle_verification=False,  # Optional but recommended
        allowed_providers=[],
        allowed_models=[]
    )
    
    if config:
        # Override with provided config
        secure_config.fail_closed = config.fail_closed
        secure_config.require_deployment_id = config.require_deployment_id
        secure_config.require_model_digest = config.require_model_digest
        secure_config.require_bundle_verification = config.require_bundle_verification
        secure_config.allowed_providers = config.allowed_providers
        secure_config.allowed_models = config.allowed_models
        if config.max_token_age is not None:
            secure_config.max_token_age = config.max_token_age
    
    return PolicyEnforcer(client, secure_config)


def create_lenient_policy(
    client: ModelSignatureClient, 
    config: Optional[PolicyConfig] = None
) -> PolicyEnforcer:
    """
    Create a policy enforcer with lenient defaults for development/testing.
    
    Args:
        client: ModelSignature client
        config: Additional policy configuration
        
    Returns:
        PolicyEnforcer configured for development use
    """
    lenient_config = PolicyConfig(
        fail_closed=False,
        require_deployment_id=False,
        require_model_digest=False,
        max_token_age=3600,  # 1 hour for development
        require_bundle_verification=False,
        allowed_providers=[],
        allowed_models=[]
    )
    
    if config:
        # Override with provided config
        lenient_config.fail_closed = config.fail_closed
        lenient_config.require_deployment_id = config.require_deployment_id
        lenient_config.require_model_digest = config.require_model_digest
        lenient_config.require_bundle_verification = config.require_bundle_verification
        lenient_config.allowed_providers = config.allowed_providers
        lenient_config.allowed_models = config.allowed_models
        if config.max_token_age is not None:
            lenient_config.max_token_age = config.max_token_age
    
    return PolicyEnforcer(client, lenient_config)