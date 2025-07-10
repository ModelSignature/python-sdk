import re
from typing import List
from dataclasses import dataclass


@dataclass
class IdentityPattern:
    """Represents an identity question pattern."""

    pattern: str
    regex: re.Pattern
    category: str
    confidence: float


class IdentityQuestionDetector:
    """Detects if user input is asking about AI identity."""

    def __init__(self, custom_patterns: List[str] | None = None):
        self.patterns = self._load_default_patterns()
        if custom_patterns:
            self.add_patterns(custom_patterns)

    def add_patterns(self, patterns: List[str]) -> None:
        for p in patterns:
            self.patterns.append(
                IdentityPattern(
                    pattern=p,
                    regex=re.compile(p, re.IGNORECASE),
                    category="custom",
                    confidence=0.8,
                )
            )

    def is_identity_question(self, text: str, threshold: float = 0.7) -> bool:
        return self.get_confidence(text) >= threshold

    def get_confidence(self, text: str) -> float:
        text = text.lower()
        score = 0.0
        for pat in self.patterns:
            if pat.regex.search(text):
                score = max(score, pat.confidence)
        return score

    def _load_default_patterns(self) -> List[IdentityPattern]:
        patterns = [
            ("who are you", r"\bwho\s+are\s+you\b", "direct", 0.95),
            ("what are you", r"\bwhat\s+are\s+you\b", "direct", 0.95),
            (
                "what's your name",
                r"\bwhat(?:'s|'s|\s+is)\s+your\s+name\b",
                "direct",
                0.9,
            ),
            ("are you gpt", r"\bare\s+you\s+gpt", "model_specific", 0.9),
            ("are you claude", r"\bare\s+you\s+claude", "model_specific", 0.9),
            (
                "which model",
                r"\bwhich\s+(?:ai\s+)?model\b",
                "model_specific",
                0.85,
            ),
            (
                "what can you do",
                r"\bwhat\s+can\s+you\s+do\b",
                "capability",
                0.7,
            ),
            (
                "your capabilities",
                r"\byour\s+capabilities\b",
                "capability",
                0.75,
            ),
            (
                "verify yourself",
                r"\bverify\s+yourself\b",
                "verification",
                0.95,
            ),
            (
                "prove you are",
                r"\bprove\s+(?:you(?:'re|'re|\s+are)|that)\b",
                "verification",
                0.9,
            ),
        ]
        return [
            IdentityPattern(p[0], re.compile(p[1], re.IGNORECASE), p[2], p[3])
            for p in patterns
        ]
