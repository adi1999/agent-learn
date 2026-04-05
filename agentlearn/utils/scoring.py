"""Shared scoring logic for eval cases."""

from __future__ import annotations

from typing import Callable, Optional

from ..models import EvalCase, Trace


def score_output(
    output: str,
    case: EvalCase,
    signal=None,
    scorer: Optional[Callable[[str, str], float]] = None,
) -> float:
    """Score an agent output against an eval case.

    Priority:
    1. If case.scorer exists: use case.scorer(output, case.task_input)
    2. If global scorer provided: use scorer(output, case.task_input)
    3. If case.expected_output exists: exact match (1.0), fuzzy/contains (0.8), miss (0.0)
    4. If signal (OutcomeSignal) provided: use signal.evaluate()
    5. Default: 0.5
    """
    # Case-level custom scorer
    if getattr(case, "scorer", None) is not None:
        return case.scorer(output, case.task_input)

    # Global scorer fallback
    if scorer is not None:
        return scorer(output, case.task_input)

    if case.expected_output is not None:
        if output.strip() == case.expected_output.strip():
            return 1.0
        if case.expected_output.strip().lower() in output.strip().lower():
            return 0.8
        return 0.0

    if signal is not None:
        trace = Trace(
            task_input=case.task_input,
            final_output=output,
        )
        outcome = signal.evaluate(trace)
        return outcome.score if outcome.score is not None else 0.5

    return 0.5
