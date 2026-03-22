"""Budget controller — manages learning spend within defined limits."""

from __future__ import annotations

from typing import Optional

from ..models import OutcomeStatus, Trace
from .logging import get_logger

logger = get_logger("budget")


class BudgetController:
    """Manages learning spend with daily and per-cycle limits.

    Prioritizes traces for analysis based on learning signal value:
    1. Failures (most learning signal)
    2. Partial successes (some signal)
    3. Successes dissimilar to eval cases (eval set growth)
    """

    def __init__(
        self,
        budget_per_day: Optional[float] = None,
        budget_per_cycle: Optional[float] = None,
        cost_per_trace_estimate: float = 0.05,
    ):
        self.budget_per_day = budget_per_day
        self.budget_per_cycle = budget_per_cycle
        self.cost_per_trace = cost_per_trace_estimate
        self._spent_today = 0.0
        self._spent_cycle = 0.0

    def can_analyze(self, estimated_cost: Optional[float] = None) -> bool:
        """Check if we can afford to analyze another trace."""
        cost = estimated_cost or self.cost_per_trace

        if self.budget_per_day is not None:
            if self._spent_today + cost > self.budget_per_day:
                return False

        if self.budget_per_cycle is not None:
            if self._spent_cycle + cost > self.budget_per_cycle:
                return False

        return True

    def record_spend(self, cost: float) -> None:
        self._spent_today += cost
        self._spent_cycle += cost

    def reset_cycle(self) -> None:
        self._spent_cycle = 0.0

    def reset_day(self) -> None:
        self._spent_today = 0.0

    def prioritize_traces(self, traces: list[Trace], budget: Optional[float] = None) -> list[Trace]:
        """Pick the most valuable traces within budget.

        Priority: failures > partials > successes
        Fills budget greedily.
        """
        limit = budget or self.budget_per_cycle
        if limit is None:
            return traces  # No budget limit

        max_traces = int(limit / self.cost_per_trace) if self.cost_per_trace > 0 else len(traces)

        def priority_key(t: Trace) -> int:
            if t.outcome is None:
                return 2
            if t.outcome.status == OutcomeStatus.FAILURE:
                return 0  # Highest priority
            if t.outcome.status == OutcomeStatus.PARTIAL:
                return 1
            return 2  # Success — lowest

        sorted_traces = sorted(traces, key=priority_key)
        return sorted_traces[:max_traces]

    @property
    def remaining_day(self) -> Optional[float]:
        if self.budget_per_day is None:
            return None
        return max(0.0, self.budget_per_day - self._spent_today)

    @property
    def remaining_cycle(self) -> Optional[float]:
        if self.budget_per_cycle is None:
            return None
        return max(0.0, self.budget_per_cycle - self._spent_cycle)
