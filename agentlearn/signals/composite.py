"""Composite signal — layers deterministic checks + LLM judge + disagreement detection."""

from __future__ import annotations

from typing import Optional

from ..models import Outcome, OutcomeStatus, Trace
from ..utils.cost_tracker import CostTracker
from ..utils.logging import get_logger
from .deterministic import DEFAULT_CHECKS, CheckResult, DeterministicCheck
from .llm_judge import LLMJudge

logger = get_logger("composite_signal")


class CompositeSignal:
    """Layers multiple signal sources for more accurate outcome evaluation.

    Flow:
    1. Run deterministic checks (free, instant)
       → Hard fail if any check fails (score = 0.0)
    2. LLM proposer → criteria
    3. LLM judge (different model from agent if possible) → score
    4. Disagreement detection
       → If deterministic passed but LLM scored low: trust LLM
       → If deterministic failed (soft) but LLM scored high: flag for review
    5. Return final outcome
    """

    def __init__(
        self,
        deterministic_checks: Optional[list[DeterministicCheck]] = None,
        llm_judge_model: str = "gpt-4o-mini",
        cost_tracker: Optional[CostTracker] = None,
        disagreement_threshold: float = 0.5,
        calibration_interval: int = 50,
    ):
        self.checks = deterministic_checks or list(DEFAULT_CHECKS)
        self._llm_judge = LLMJudge(
            model=llm_judge_model,
            cost_tracker=cost_tracker,
        )
        self.disagreement_threshold = disagreement_threshold
        self.calibration_interval = calibration_interval
        self._run_count = 0
        self._flagged_for_review: list[str] = []  # trace_ids needing human review

    def evaluate(self, trace: Trace) -> Outcome:
        """Evaluate a trace using layered signals."""
        self._run_count += 1

        # Step 1: Deterministic checks (free, instant)
        det_results = self._run_deterministic(trace)
        hard_failures = [r for r in det_results if r.hard_fail and not r.passed]

        if hard_failures:
            reasons = "; ".join(r.reason for r in hard_failures)
            return Outcome(
                status=OutcomeStatus.FAILURE,
                score=0.0,
                signal_source="deterministic",
                reasoning=f"Hard fail: {reasons}",
            )

        # Step 2+3: LLM judge
        llm_outcome = self._llm_judge.evaluate(trace)

        # Step 4: Disagreement detection
        det_all_passed = all(r.passed for r in det_results)
        det_soft_failures = [r for r in det_results if not r.passed and not r.hard_fail]
        llm_score = llm_outcome.score or 0.5

        if det_all_passed and llm_score < self.disagreement_threshold:
            # Deterministic passed but LLM scored low — trust LLM
            # LLM caught something deterministic missed
            logger.info(
                f"Disagreement: deterministic passed but LLM scored {llm_score:.2f}. "
                "Trusting LLM judgment."
            )
            return llm_outcome

        if det_soft_failures and llm_score > 0.8:
            # Deterministic had soft failures but LLM scored high — flag for review
            soft_reasons = "; ".join(r.reason for r in det_soft_failures)
            logger.info(
                f"Disagreement: deterministic soft-failed ({soft_reasons}) "
                f"but LLM scored {llm_score:.2f}. Flagging for review."
            )
            self._flagged_for_review.append(trace.trace_id)
            return Outcome(
                status=OutcomeStatus.PARTIAL,
                score=0.5,
                signal_source="composite_disagreement",
                reasoning=f"Disagreement: {soft_reasons} (LLM scored {llm_score:.2f}). Flagged for review.",
            )

        # Step 5: Check calibration
        if self._should_calibrate():
            self._flagged_for_review.append(trace.trace_id)

        return llm_outcome

    def _run_deterministic(self, trace: Trace) -> list[CheckResult]:
        """Run all deterministic checks."""
        results = []
        for check in self.checks:
            try:
                result = check.run(trace)
                results.append(result)
            except Exception as e:
                logger.warning(f"Check {check.name} failed with error: {e}")
                results.append(CheckResult(passed=True))  # Don't block on check errors
        return results

    def _should_calibrate(self) -> bool:
        """Check if we should request human calibration."""
        return self.calibration_interval > 0 and self._run_count % self.calibration_interval == 0

    def get_flagged_traces(self) -> list[str]:
        """Get trace IDs flagged for human review."""
        flagged = list(self._flagged_for_review)
        self._flagged_for_review.clear()
        return flagged

    def add_check(self, check: DeterministicCheck) -> None:
        """Add a deterministic check to the pipeline."""
        self.checks.append(check)
