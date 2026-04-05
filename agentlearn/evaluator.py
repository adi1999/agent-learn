"""Batch evaluation — run an agent against an eval set, score, and aggregate."""

from __future__ import annotations

import asyncio
import csv
import inspect
import time
from typing import Callable

from .models import EvalCase, EvalComparison, EvalResult, EvalRunReport
from .utils.logging import get_logger
from .utils.scoring import score_output

logger = get_logger("evaluator")


class BatchEvaluator:
    """Runs an agent function against eval cases and produces an aggregate report."""

    def __init__(self, signal=None, pass_threshold: float = 0.7):
        self._signal = signal
        self.pass_threshold = pass_threshold

    def run(
        self,
        agent_func: Callable[[str], str],
        eval_cases: list[EvalCase],
        name: str = "",
    ) -> EvalRunReport:
        """Run agent_func on every eval case, score, and aggregate."""
        start_time = time.time()
        results: list[EvalResult] = []

        for case in eval_cases:
            result = self._run_single(agent_func, case)
            results.append(result)

        report = self._aggregate(results, name)
        report.duration_seconds = time.time() - start_time
        report.signal_source = type(self._signal).__name__ if self._signal else "expected_output"
        return report

    def _run_single(self, agent_func: Callable[[str], str], case: EvalCase) -> EvalResult:
        """Run and score a single eval case."""
        case_start = time.time()
        try:
            output = agent_func(case.task_input)
            score = score_output(str(output), case, signal=self._signal)
            return EvalResult(
                eval_id=case.eval_id,
                task_input=case.task_input,
                expected_output=case.expected_output,
                agent_output=str(output),
                score=score,
                passed=score >= self.pass_threshold,
                tags=case.tags,
                duration_seconds=time.time() - case_start,
            )
        except Exception as e:
            logger.warning(f"Agent error on eval {case.eval_id}: {e}")
            return EvalResult(
                eval_id=case.eval_id,
                task_input=case.task_input,
                expected_output=case.expected_output,
                agent_output="",
                score=0.0,
                passed=False,
                tags=case.tags,
                duration_seconds=time.time() - case_start,
                error=str(e),
            )

    async def run_async(
        self,
        agent_func: Callable,
        eval_cases: list[EvalCase],
        name: str = "",
    ) -> EvalRunReport:
        """Run agent_func (sync or async) on every eval case, score, and aggregate."""
        start_time = time.time()
        results: list[EvalResult] = []
        is_async = inspect.iscoroutinefunction(agent_func)

        for case in eval_cases:
            result = await self._run_single_async(agent_func, case, is_async)
            results.append(result)

        report = self._aggregate(results, name)
        report.duration_seconds = time.time() - start_time
        report.signal_source = type(self._signal).__name__ if self._signal else "expected_output"
        return report

    async def _run_single_async(
        self, agent_func: Callable, case: EvalCase, is_async: bool
    ) -> EvalResult:
        """Run and score a single eval case, supporting async agent functions."""
        case_start = time.time()
        try:
            if is_async:
                output = await agent_func(case.task_input)
            else:
                output = await asyncio.to_thread(agent_func, case.task_input)
            score = score_output(str(output), case, signal=self._signal)
            return EvalResult(
                eval_id=case.eval_id,
                task_input=case.task_input,
                expected_output=case.expected_output,
                agent_output=str(output),
                score=score,
                passed=score >= self.pass_threshold,
                tags=case.tags,
                duration_seconds=time.time() - case_start,
            )
        except Exception as e:
            logger.warning(f"Agent error on eval {case.eval_id}: {e}")
            return EvalResult(
                eval_id=case.eval_id,
                task_input=case.task_input,
                expected_output=case.expected_output,
                agent_output="",
                score=0.0,
                passed=False,
                tags=case.tags,
                duration_seconds=time.time() - case_start,
                error=str(e),
            )

    def _aggregate(self, results: list[EvalResult], name: str) -> EvalRunReport:
        """Aggregate individual results into a report."""
        if not results:
            return EvalRunReport(name=name, pass_threshold=self.pass_threshold)

        scores = [r.score for r in results]
        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.error is not None)
        failed = len(results) - passed - errored

        # Per-tag breakdown
        tag_stats: dict[str, dict] = {}
        for r in results:
            for tag in r.tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "passed": 0, "scores": []}
                tag_stats[tag]["total"] += 1
                tag_stats[tag]["scores"].append(r.score)
                if r.passed:
                    tag_stats[tag]["passed"] += 1

        # Compute tag averages
        for tag, stats in tag_stats.items():
            tag_scores = stats.pop("scores")
            stats["accuracy"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_score"] = sum(tag_scores) / len(tag_scores) if tag_scores else 0.0

        return EvalRunReport(
            name=name,
            total_cases=len(results),
            passed=passed,
            failed=failed,
            errored=errored,
            accuracy=passed / len(results) if results else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            tag_stats=tag_stats,
            results=results,
            pass_threshold=self.pass_threshold,
        )

    @staticmethod
    def import_csv(path: str) -> list[EvalCase]:
        """Import eval cases from a CSV file.

        Expected columns: task_input (required), expected_output, tags (comma-separated), judge_prompt
        """
        cases: list[EvalCase] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_input = row.get("task_input", "").strip()
                if not task_input:
                    continue

                expected = row.get("expected_output", "").strip() or None
                judge_prompt = row.get("judge_prompt", "").strip() or None
                tags_raw = row.get("tags", "").strip()
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

                cases.append(
                    EvalCase(
                        task_input=task_input,
                        expected_output=expected,
                        judge_prompt=judge_prompt,
                        tags=tags,
                        source="csv_import",
                    )
                )
        return cases

    @staticmethod
    def import_dicts(data: list[dict]) -> list[EvalCase]:
        """Convert a list of dicts to EvalCase objects.

        Each dict should have at minimum: {"task_input": "..."}.
        Optional keys: expected_output, judge_prompt, tags (list of strings).
        """
        cases: list[EvalCase] = []
        for d in data:
            task_input = d.get("task_input", "").strip()
            if not task_input:
                continue

            expected = d.get("expected_output")
            if isinstance(expected, str):
                expected = expected.strip() or None

            judge_prompt = d.get("judge_prompt")
            if isinstance(judge_prompt, str):
                judge_prompt = judge_prompt.strip() or None

            tags = d.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            cases.append(
                EvalCase(
                    task_input=task_input,
                    expected_output=expected,
                    judge_prompt=judge_prompt,
                    tags=tags if isinstance(tags, list) else [],
                    source="dict_import",
                )
            )
        return cases

    @staticmethod
    def compare(baseline: EvalRunReport, treatment: EvalRunReport) -> EvalComparison:
        """Compare two eval runs, matching cases by task_input."""
        baseline_map = {r.task_input: r for r in baseline.results}
        treatment_map = {r.task_input: r for r in treatment.results}

        regressions = []
        improvements = []

        common_keys = set(baseline_map.keys()) & set(treatment_map.keys())
        for key in common_keys:
            b = baseline_map[key]
            t = treatment_map[key]
            if t.score < b.score:
                regressions.append(
                    {
                        "task_input": key[:100],
                        "baseline_score": b.score,
                        "treatment_score": t.score,
                        "delta": t.score - b.score,
                    }
                )
            elif t.score > b.score:
                improvements.append(
                    {
                        "task_input": key[:100],
                        "baseline_score": b.score,
                        "treatment_score": t.score,
                        "delta": t.score - b.score,
                    }
                )

        return EvalComparison(
            baseline_run_id=baseline.run_id,
            treatment_run_id=treatment.run_id,
            baseline_accuracy=baseline.accuracy,
            treatment_accuracy=treatment.accuracy,
            accuracy_delta=treatment.accuracy - baseline.accuracy,
            baseline_avg_score=baseline.avg_score,
            treatment_avg_score=treatment.avg_score,
            score_delta=treatment.avg_score - baseline.avg_score,
            regressions=regressions,
            improvements=improvements,
        )
