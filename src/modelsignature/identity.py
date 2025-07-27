import re
from typing import List, Optional, Dict
from difflib import SequenceMatcher
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class IdentityPattern:
    """Represents an identity question pattern."""

    pattern: str
    regex: re.Pattern
    category: str
    confidence: float


PATTERN_CATEGORIES: Dict[str, List[str]] = {
    "direct_identity": [
        "who are you",
        "who you are",
        "what are you",
        "what's your name",
        "what is your name",
        "tell me about yourself",
    ],
    "model_specific": [
        "are you gpt",
        "are you claude",
        "are you chatgpt",
        "which model are you",
        "what model is this",
        "which ai am i talking to",
    ],
    "capability": [
        "what can you do",
        "what are your capabilities",
        "what are you capable of",
        "how do you work",
    ],
    "verification": [
        "prove who you are",
        "verify yourself",
        "how do i know you're real",
        "are you really",
        "confirm your identity",
    ],
}

MULTILINGUAL_PATTERNS: Dict[str, List[str]] = {
    "french": ["qui êtes-vous", "qu'est-ce que vous êtes"],
    "spanish": ["quién eres", "qué eres"],
    "german": ["wer bist du", "was bist du"],
    "russian": ["кто ты"],
}


class IdentityQuestionDetector:
    """Detects if user input is asking about AI identity."""

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        self.patterns = self._load_default_patterns()
        self.multilingual = self._load_multilingual_patterns()
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
        if self._quick_pattern_match(text):
            return True
        normalized = self._normalize_text(text)
        if self._normalized_pattern_match(normalized):
            return True
        return self._fuzzy_match(text, threshold)

    def get_confidence(self, text: str) -> float:
        normalized = self._normalize_text(text)
        score = 0.0
        for pat in self.patterns:
            if pat.regex.search(normalized):
                score = max(score, pat.confidence)
        for lang_pats in self.multilingual.values():
            for rp in lang_pats:
                if rp.search(normalized):
                    score = max(score, 0.9)
        if score < 0.8 and self._fuzzy_match(text, 0.8):
            score = max(score, 0.8)
        return score

    def _quick_pattern_match(self, text: str) -> bool:
        for pat in self.patterns:
            if pat.regex.search(text):
                return True
        for lang_pats in self.multilingual.values():
            for rp in lang_pats:
                if rp.search(text):
                    return True
        return False

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        contractions = {
            "what's": "what is",
            "who's": "who is",
            "you're": "you are",
            "i'm": "i am",
            "can't": "cannot",
            "it's": "it is",
        }
        for c, repl in contractions.items():
            text = text.replace(c, repl)
        replacements = {
            r"\bu\b": "you",
            r"\br\b": "are",
            r"\bya\b": "you",
            r"wat": "what",
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalized_pattern_match(self, text: str) -> bool:
        for pat in self.patterns:
            if pat.regex.search(text):
                return True
        for lang_pats in self.multilingual.values():
            for rp in lang_pats:
                if rp.search(text):
                    return True
        return False

    def _fuzzy_match(self, text: str, threshold: float) -> bool:
        cmp = text.lower()
        patterns = [p.pattern for p in self.patterns]
        for lang_pats in self.multilingual.values():
            patterns.extend([rp.pattern for rp in lang_pats])

        @lru_cache(maxsize=256)
        def _ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

        for pat in patterns:
            if _ratio(cmp, pat) >= threshold:
                return True
        return False

    def _load_default_patterns(self) -> List[IdentityPattern]:
        patterns = []
        for category, pats in PATTERN_CATEGORIES.items():
            for p in pats:
                regex = re.compile(p, re.IGNORECASE)
                confidence = 0.9 if category == "direct_identity" else 0.85
                patterns.append((p, regex, category, confidence))

        return [IdentityPattern(p[0], p[1], p[2], p[3]) for p in patterns]

    def _load_multilingual_patterns(self) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for lang, pats in MULTILINGUAL_PATTERNS.items():
            compiled[lang] = [
                re.compile(p, re.IGNORECASE) for p in pats
            ]  # noqa: E501
        return compiled
