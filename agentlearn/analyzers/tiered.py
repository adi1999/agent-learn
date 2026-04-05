"""Tiered analyzer — cheap heuristics first, expensive LLM only when warranted."""

from __future__ import annotations

import re
from typing import Optional

from ..models import CandidateFix, FixType, Trace
from ..utils.cost_tracker import CostTracker
from ..utils.logging import get_logger

logger = get_logger("tiered_analyzer")

# Known failure patterns that don't need LLM analysis
KNOWN_PATTERNS = [
    (
        r"json\.?decode|invalid json|json parse",
        "json_parsing",
        "Validate JSON structure before parsing. Check for empty responses, trailing commas, and unescaped characters.",
    ),
    (
        r"timeout|timed?\s*out|deadline exceeded",
        "timeout",
        "Add timeout handling with exponential backoff. Set explicit timeouts on all external calls.",
    ),
    (
        r"rate.?limit|429|too many requests",
        "rate_limit",
        "Implement rate limiting with backoff. Check rate limit headers before retrying.",
    ),
    (
        r"auth|unauthorized|403|401|forbidden",
        "auth_failure",
        "Verify credentials before making API calls. Check token expiration and refresh if needed.",
    ),
    (
        r"connection.?(refused|reset|error)|network",
        "network_error",
        "Add retry logic for transient network errors. Verify endpoint availability before requests.",
    ),
    (
        r"null|none|undefined|missing.?field",
        "null_handling",
        "Check for null/missing values before processing. Provide sensible defaults for optional fields.",
    ),
]

_compiled_known = [(re.compile(p, re.IGNORECASE), name, fix) for p, name, fix in KNOWN_PATTERNS]


class TieredAnalyzer:
    """Cheap heuristics first, expensive LLM analysis only when warranted.

    Tier 1: Pattern matching against known failure types (free, instant)
    Tier 2: Full LLM analysis (expensive, thorough)

    Saves LLM costs by handling common failures with pre-built fixes.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.model = model
        self.cost_tracker = cost_tracker
        self._llm_analyzer = None  # Lazy init

    def analyze_single(self, trace: Trace) -> list[CandidateFix]:
        """Analyze a trace using tiered approach."""
        # Tier 1: Pattern matching (free)
        result = self._quick_classify(trace)
        if result is not None:
            return result

        # Tier 2: Full LLM analysis (expensive)
        return self._full_analysis(trace)

    def analyze_batch(self, traces: list[Trace]) -> list[CandidateFix]:
        """Analyze batch with tiered approach."""
        all_candidates = []
        for trace in traces:
            all_candidates.extend(self.analyze_single(trace))
        return all_candidates

    def _quick_classify(self, trace: Trace) -> Optional[list[CandidateFix]]:
        """Tier 1: Check against known failure patterns."""
        # Build searchable text from trace
        text_parts = [trace.task_input or "", trace.final_output or ""]
        if trace.outcome:
            text_parts.append(trace.outcome.reasoning or "")
        for step in trace.steps:
            text_parts.append(step.result or "")
        text = " ".join(text_parts)

        for compiled, pattern_name, fix_content in _compiled_known:
            if compiled.search(text):
                logger.info(f"Tier 1 match: {pattern_name} for trace {trace.trace_id[:8]}")
                return [
                    CandidateFix(
                        fix_type=FixType.CHECKLIST,
                        content=fix_content,
                        applies_when=f"When the agent encounters {pattern_name.replace('_', ' ')} issues",
                        source_trace_ids=[trace.trace_id],
                        confidence=0.7,
                        reasoning=f"Known pattern match: {pattern_name}",
                    )
                ]

        return None  # No known pattern — proceed to Tier 2

    def _full_analysis(self, trace: Trace) -> list[CandidateFix]:
        """Tier 2: Full LLM analysis."""
        if self._llm_analyzer is None:
            from .llm_analyzer import LLMAnalyzer

            self._llm_analyzer = LLMAnalyzer(model=self.model, cost_tracker=self.cost_tracker)
        return self._llm_analyzer.analyze_single(trace)
