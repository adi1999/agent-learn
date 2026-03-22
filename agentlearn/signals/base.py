"""OutcomeSignal protocol — determines if a run succeeded."""

from typing import Protocol, runtime_checkable

from ..models import Outcome, Trace


@runtime_checkable
class OutcomeSignal(Protocol):
    def evaluate(self, trace: Trace) -> Outcome:
        """Determine if a run succeeded, failed, or partially succeeded."""
        ...
