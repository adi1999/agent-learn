"""Analyzer protocol — extracts lessons from traces."""

from typing import Protocol, runtime_checkable

from ..models import CandidateFix, Trace


@runtime_checkable
class Analyzer(Protocol):
    def analyze_single(self, trace: Trace) -> list[CandidateFix]:
        """Analyze one failed trace. Return candidate fixes."""
        ...

    def analyze_batch(self, traces: list[Trace]) -> list[CandidateFix]:
        """Find patterns across multiple traces. Return candidate fixes."""
        ...
