"""Statistical validator — A/B tests candidate fixes against held-out eval sets."""

from __future__ import annotations

import math
from typing import Callable, Optional

from ..models import (
    CandidateFix,
    EvalCase,
    EvalCaseResult,
    ValidationResult,
)
from ..utils.logging import get_logger

logger = get_logger("statistical_validator")

# Conservative thresholds
NOISE_MARGIN = 0.05
MAX_REGRESSIONS = 0
MIN_CONFIDENCE = 0.8
MIN_EVAL_CASES = 5


def _welchs_t_test(a: list[float], b: list[float]) -> float:
    """Compute confidence that b > a using Welch's t-test.

    Returns a confidence level between 0.0 and 1.0.
    Higher values = more confident that b is better than a.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    # When variance is zero (all values identical), result is deterministic
    if var_a == 0 and var_b == 0:
        if mean_b > mean_a:
            return 1.0
        elif mean_b == mean_a:
            return 0.5
        else:
            return 0.0

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 1.0 if mean_b > mean_a else 0.0

    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom_parts = []
    if var_a > 0:
        denom_parts.append((var_a / n_a) ** 2 / (n_a - 1))
    if var_b > 0:
        denom_parts.append((var_b / n_b) ** 2 / (n_b - 1))
    denom = sum(denom_parts)
    df = num / denom if denom > 0 else 1.0

    return _t_to_confidence(t_stat, df)


def _t_to_confidence(t_stat: float, df: float) -> float:
    """Convert t-statistic to confidence level.

    Uses a logistic approximation to the t-distribution CDF.
    Returns P(T < t) — the probability that b > a.
    """
    if df <= 0:
        return 0.5

    # Logistic approximation to t-distribution CDF
    # P(T < t) ≈ 1 / (1 + exp(-k*t)) where k depends on df
    k = math.sqrt(2 / math.pi) * (1 + 1 / (4 * max(df, 1)))
    try:
        confidence = 1.0 / (1.0 + math.exp(-k * t_stat))
    except OverflowError:
        confidence = 0.0 if t_stat < 0 else 1.0

    return max(0.0, min(1.0, confidence))


class StatisticalValidator:
    """Validates candidate fixes by A/B testing against held-out eval cases.

    Conservative by design:
    - Fix must improve by at least NOISE_MARGIN (5%)
    - Zero regressions allowed
    - Statistical confidence must exceed MIN_CONFIDENCE (80%)
    - Falls back to human approval if eval set is too small
    """

    def __init__(
        self,
        noise_margin: float = NOISE_MARGIN,
        max_regressions: int = MAX_REGRESSIONS,
        min_confidence: float = MIN_CONFIDENCE,
        min_eval_cases: int = MIN_EVAL_CASES,
        outcome_signal=None,
        approval_callback: Optional[Callable[[CandidateFix], bool]] = None,
    ):
        self.noise_margin = noise_margin
        self.max_regressions = max_regressions
        self.min_confidence = min_confidence
        self.min_eval_cases = min_eval_cases
        self._signal = outcome_signal
        self._approval_callback = approval_callback

    def validate(
        self,
        candidate: CandidateFix,
        eval_set: list[EvalCase],
        agent_runner: Callable[[str, list[str]], str],
        approval_callback: Optional[Callable[[CandidateFix], bool]] = None,
    ) -> ValidationResult:
        """Test a fix against held-out eval cases."""
        callback = approval_callback or self._approval_callback

        if len(eval_set) < self.min_eval_cases:
            logger.info(
                f"Only {len(eval_set)} eval cases (need {self.min_eval_cases}). "
                "Deferring to human approval."
            )
            return self._defer_to_human(candidate, callback)

        baseline_scores: list[float] = []
        treatment_scores: list[float] = []
        details: list[EvalCaseResult] = []
        total_cost = 0.0

        for case in eval_set:
            # Run WITHOUT the fix (baseline)
            try:
                baseline_output = agent_runner(case.task_input, [])
                baseline_score = self._score_output(baseline_output, case)
            except Exception as e:
                logger.warning(f"Baseline run failed for eval {case.eval_id}: {e}")
                baseline_score = 0.0

            # Run WITH the fix (treatment)
            try:
                treatment_output = agent_runner(case.task_input, [candidate.fix_id])
                treatment_score = self._score_output(treatment_output, case)
            except Exception as e:
                logger.warning(f"Treatment run failed for eval {case.eval_id}: {e}")
                treatment_score = 0.0

            baseline_scores.append(baseline_score)
            treatment_scores.append(treatment_score)

            passed = treatment_score >= baseline_score
            details.append(
                EvalCaseResult(
                    task_input=case.task_input,
                    baseline_score=baseline_score,
                    treatment_score=treatment_score,
                    passed=passed,
                )
            )

        # Compute statistics
        improvement = _mean(treatment_scores) - _mean(baseline_scores)
        regressions = sum(1 for t, b in zip(treatment_scores, baseline_scores) if t < b)
        confidence = _welchs_t_test(baseline_scores, treatment_scores)

        passed = (
            improvement > self.noise_margin
            and regressions <= self.max_regressions
            and confidence >= self.min_confidence
        )

        logger.info(
            f"Validation: improvement={improvement:.3f}, regressions={regressions}, "
            f"confidence={confidence:.3f}, passed={passed}"
        )

        return ValidationResult(
            passed=passed,
            improvement=improvement,
            regression_count=regressions,
            confidence=confidence,
            details=details,
            cost_usd=total_cost,
        )

    def _score_output(self, output: str, case: EvalCase) -> float:
        """Score an output against an eval case."""
        from ..utils.scoring import score_output

        return score_output(output, case, signal=self._signal)

    def _defer_to_human(
        self,
        candidate: CandidateFix,
        callback: Optional[Callable[[CandidateFix], bool]],
    ) -> ValidationResult:
        """Fall back to human approval when eval set is too small."""
        if callback is not None:
            approved = callback(candidate)
            return ValidationResult(
                passed=approved,
                confidence=1.0 if approved else 0.0,
            )

        # No callback — reject by default (safe)
        logger.warning("No approval callback and eval set too small. Rejecting.")
        return ValidationResult(passed=False, confidence=0.0)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
