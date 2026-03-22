"""Tracer protocol — observes agent execution."""

from typing import Protocol, runtime_checkable

from ..models import Outcome, Step, Trace


@runtime_checkable
class Tracer(Protocol):
    def start_trace(self, task_input: str, metadata: dict = {}) -> str:
        """Begin tracing a new agent run. Returns trace_id."""
        ...

    def record_step(self, trace_id: str, step: Step) -> None:
        """Record a single decision/action within a run."""
        ...

    def end_trace(self, trace_id: str, outcome: Outcome, final_output: str = "") -> Trace:
        """Finalize the trace with outcome signal. Returns complete Trace."""
        ...
