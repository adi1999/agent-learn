"""Validator protocol — tests candidate fixes before promotion."""

from typing import Callable, Optional, Protocol, runtime_checkable

from ..models import CandidateFix, EvalCase, ValidationResult


@runtime_checkable
class Validator(Protocol):
    def validate(
        self,
        candidate: CandidateFix,
        eval_set: list[EvalCase],
        agent_runner: Callable[[str, list[str]], str],
        approval_callback: Optional[Callable[[CandidateFix], bool]] = None,
    ) -> ValidationResult:
        """Test a fix against held-out problems. Returns validation result.

        approval_callback: Optional callback for human approval. Receives the
        candidate and returns True (approve) or False (reject). When None,
        the validator uses its default approval mechanism (e.g., CLI prompt).
        """
        ...
