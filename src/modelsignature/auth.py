from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AuthHandler:
    """Simple auth handler to store API key."""

    api_key: Optional[str] = None

    def apply(self, headers: dict) -> None:
        if self.api_key:
            headers["X-API-Key"] = self.api_key
