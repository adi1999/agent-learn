"""Content safety checks for knowledge items before injection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import KnowledgeItem
from .logging import get_logger

logger = get_logger("safety")

# Regex threat patterns — catches obvious attacks without LLM cost
THREAT_PATTERNS = [
    # Instruction override
    (r"ignore\s+(previous|all|above|prior)(\s+\w+)*\s+instructions", "instruction_override"),
    (r"disregard\s+(your|the)\s+(system|original)\s+prompt", "instruction_override"),
    (r"you\s+are\s+now\s+[A-Z]", "persona_hijack"),
    (r"forget\s+(everything|all|your)\s+(you|instructions|rules)", "instruction_override"),
    (r"new\s+instructions?\s*:", "instruction_override"),
    # Exfiltration
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASS)", "exfiltration"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.ssh|id_rsa)", "exfiltration"),
    (r"send\s+(to|data|this)\s+to\s+https?://", "exfiltration"),
    (r"wget\s+.*\|.*sh", "exfiltration"),
    (r"base64\s+.*\|\s*(bash|sh|eval)", "exfiltration"),
    # Destructive
    (r"rm\s+-rf\s+/", "destructive"),
    (r"drop\s+table", "destructive"),
    (r"delete\s+all", "destructive"),
    (r"truncate\s+table", "destructive"),
    (r"format\s+[cC]:", "destructive"),
    # Deception
    (r"do\s+not\s+tell\s+the\s+user", "deception"),
    (r"hide\s+this\s+from", "deception"),
    (r"pretend\s+(you|that|this)", "deception"),
    (r"never\s+reveal\s+(this|that|your)", "deception"),
]

# Invisible unicode characters used in injection attacks
INVISIBLE_CHARS = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
    "\ufeff",  # Zero-width no-break space (BOM)
    "\u00ad",  # Soft hyphen
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
    "\u202a",  # Left-to-right embedding
    "\u202b",  # Right-to-left embedding
    "\u202c",  # Pop directional formatting
    "\u2062",  # Invisible times
    "\u2063",  # Invisible separator
}

_compiled_patterns = [(re.compile(p, re.IGNORECASE), cat) for p, cat in THREAT_PATTERNS]


@dataclass
class SafetyResult:
    safe: bool
    threats: list[dict] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.threats:
            return ""
        return "; ".join(t["reason"] for t in self.threats)


class ContentSafetyCheck:
    """Scans knowledge items for potential threats before injection.

    Uses regex patterns (free, instant) to catch obvious attacks:
    - Instruction override attempts
    - Data exfiltration commands
    - Destructive operations
    - Deception instructions
    - Invisible unicode injection characters
    """

    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None):
        self._patterns = list(_compiled_patterns)
        if extra_patterns:
            for pattern, category in extra_patterns:
                self._patterns.append((re.compile(pattern, re.IGNORECASE), category))

    def check(self, item: KnowledgeItem) -> SafetyResult:
        """Check a knowledge item for safety threats."""
        threats = []

        text = f"{item.content} {item.applies_when}"

        # Pattern matching
        for compiled, category in self._patterns:
            match = compiled.search(text)
            if match:
                threats.append({
                    "category": category,
                    "matched": match.group(),
                    "reason": f"Blocked pattern ({category}): '{match.group()}'",
                })

        # Invisible character detection
        invisible_found = []
        for char in INVISIBLE_CHARS:
            if char in text:
                invisible_found.append(repr(char))

        if invisible_found:
            threats.append({
                "category": "invisible_chars",
                "matched": ", ".join(invisible_found),
                "reason": f"Invisible unicode characters detected: {', '.join(invisible_found)}",
            })

        return SafetyResult(safe=len(threats) == 0, threats=threats)

    def check_batch(self, items: list[KnowledgeItem]) -> dict[str, SafetyResult]:
        """Check multiple items. Returns {item_id: SafetyResult}."""
        return {item.item_id: self.check(item) for item in items}

    def filter_safe(self, items: list[KnowledgeItem]) -> list[KnowledgeItem]:
        """Return only items that pass safety checks."""
        safe_items = []
        for item in items:
            result = self.check(item)
            if result.safe:
                safe_items.append(item)
            else:
                logger.warning(
                    f"Blocked unsafe knowledge item {item.item_id[:8]}: {result.reason}"
                )
        return safe_items
