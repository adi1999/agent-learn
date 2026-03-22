"""Validator protocol — tests candidate fixes before promotion."""

from typing import Callable, Protocol, runtime_checkable

from ..models import CandidateFix, EvalCase, ValidationResult


@runtime_checkable
class Validator(Protocol):
    def validate(
        self,
        candidate: CandidateFix,
        eval_set: list[EvalCase],
        agent_runner: Callable[[str, list[str]], str],
    ) -> ValidationResult:
        """Test a fix against held-out problems. Returns validation result."""
        ...
