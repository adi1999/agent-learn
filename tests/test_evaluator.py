"""Tests for batch evaluation: scoring, import, evaluator, store, comparison, CLI."""

import csv
import os
from unittest.mock import patch

import pytest

from agentlearn.evaluator import BatchEvaluator
from agentlearn.models import (
    EvalCase,
    EvalComparison,
    EvalResult,
    EvalRunReport,
    Outcome,
    OutcomeStatus,
    Trace,
)
from agentlearn.store.local_store import LocalStore
from agentlearn.utils.scoring import score_output


# === Fixtures ===


@pytest.fixture
def sample_eval_cases():
    """A set of eval cases with tags and expected outputs."""
    return [
        EvalCase(
            task_input="What is 2+2?",
            expected_output="4",
            tags=["math", "easy"],
            source="manual",
        ),
        EvalCase(
            task_input="What is 10*5?",
            expected_output="50",
            tags=["math", "easy"],
            source="manual",
        ),
        EvalCase(
            task_input="What is the capital of France?",
            expected_output="Paris",
            tags=["geography"],
            source="manual",
        ),
        EvalCase(
            task_input="What is 100/4?",
            expected_output="25",
            tags=["math", "medium"],
            source="manual",
        ),
        EvalCase(
            task_input="Who wrote Hamlet?",
            expected_output="Shakespeare",
            tags=["literature"],
            source="manual",
        ),
    ]


def good_agent(task: str) -> str:
    """An agent that returns correct answers."""
    answers = {
        "What is 2+2?": "4",
        "What is 10*5?": "50",
        "What is the capital of France?": "Paris",
        "What is 100/4?": "25",
        "Who wrote Hamlet?": "Shakespeare",
    }
    return answers.get(task, "I don't know")


def bad_agent(task: str) -> str:
    """An agent that always returns wrong answers."""
    return "wrong answer"


def partial_agent(task: str) -> str:
    """An agent that gets some right and some wrong."""
    if "+" in task or "*" in task:
        return good_agent(task)
    return "I don't know"


def error_agent(task: str) -> str:
    """An agent that raises exceptions."""
    raise ValueError("Agent crashed")


# === TestScoreOutput ===


class TestScoreOutput:
    def test_exact_match(self):
        case = EvalCase(task_input="test", expected_output="hello")
        assert score_output("hello", case) == 1.0

    def test_exact_match_strips_whitespace(self):
        case = EvalCase(task_input="test", expected_output="hello")
        assert score_output("  hello  ", case) == 1.0

    def test_fuzzy_match(self):
        case = EvalCase(task_input="test", expected_output="Paris")
        assert score_output("The answer is Paris, the capital of France", case) == 0.8

    def test_fuzzy_match_case_insensitive(self):
        case = EvalCase(task_input="test", expected_output="Paris")
        assert score_output("the answer is paris", case) == 0.8

    def test_no_match(self):
        case = EvalCase(task_input="test", expected_output="Paris")
        assert score_output("London", case) == 0.0

    def test_signal_based_scoring(self):
        """When no expected_output, use OutcomeSignal."""

        class MockSignal:
            def evaluate(self, trace: Trace) -> Outcome:
                return Outcome(status=OutcomeStatus.SUCCESS, score=0.85)

        case = EvalCase(task_input="test")
        assert score_output("any output", case, signal=MockSignal()) == 0.85

    def test_signal_returns_none_score(self):
        class MockSignal:
            def evaluate(self, trace: Trace) -> Outcome:
                return Outcome(status=OutcomeStatus.PARTIAL, score=None)

        case = EvalCase(task_input="test")
        assert score_output("any output", case, signal=MockSignal()) == 0.5

    def test_default_score_no_expected_no_signal(self):
        case = EvalCase(task_input="test")
        assert score_output("any output", case) == 0.5


# === TestBatchImport ===


class TestBatchImport:
    def test_import_csv_all_columns(self, tmp_path):
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_input", "expected_output", "tags", "judge_prompt"])
            writer.writerow(["What is 2+2?", "4", "math,easy", "Check the answer"])
            writer.writerow(["Capital of France?", "Paris", "geography", ""])

        cases = BatchEvaluator.import_csv(str(csv_path))
        assert len(cases) == 2
        assert cases[0].task_input == "What is 2+2?"
        assert cases[0].expected_output == "4"
        assert cases[0].tags == ["math", "easy"]
        assert cases[0].judge_prompt == "Check the answer"
        assert cases[0].source == "csv_import"

    def test_import_csv_minimal_columns(self, tmp_path):
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_input"])
            writer.writerow(["What is 2+2?"])
            writer.writerow(["Capital of France?"])

        cases = BatchEvaluator.import_csv(str(csv_path))
        assert len(cases) == 2
        assert cases[0].expected_output is None
        assert cases[0].tags == []

    def test_import_csv_skips_empty_task(self, tmp_path):
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_input", "expected_output"])
            writer.writerow(["What is 2+2?", "4"])
            writer.writerow(["", ""])
            writer.writerow(["Capital of France?", "Paris"])

        cases = BatchEvaluator.import_csv(str(csv_path))
        assert len(cases) == 2

    def test_import_dicts_full(self):
        data = [
            {"task_input": "What is 2+2?", "expected_output": "4", "tags": ["math"]},
            {"task_input": "Capital?", "expected_output": "Paris"},
        ]
        cases = BatchEvaluator.import_dicts(data)
        assert len(cases) == 2
        assert cases[0].tags == ["math"]
        assert cases[1].tags == []
        assert cases[0].source == "dict_import"

    def test_import_dicts_minimal(self):
        data = [{"task_input": "What is 2+2?"}]
        cases = BatchEvaluator.import_dicts(data)
        assert len(cases) == 1
        assert cases[0].expected_output is None

    def test_import_dicts_skips_empty(self):
        data = [{"task_input": ""}, {"task_input": "Valid"}]
        cases = BatchEvaluator.import_dicts(data)
        assert len(cases) == 1

    def test_import_dicts_tags_as_string(self):
        data = [{"task_input": "test", "tags": "math,easy"}]
        cases = BatchEvaluator.import_dicts(data)
        assert cases[0].tags == ["math", "easy"]


# === TestBatchEvaluator ===


class TestBatchEvaluator:
    def test_perfect_agent(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        report = evaluator.run(good_agent, sample_eval_cases, name="perfect_run")

        assert report.name == "perfect_run"
        assert report.total_cases == 5
        assert report.passed == 5
        assert report.failed == 0
        assert report.errored == 0
        assert report.accuracy == 1.0
        assert report.avg_score == 1.0
        assert report.min_score == 1.0
        assert report.max_score == 1.0

    def test_bad_agent(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        report = evaluator.run(bad_agent, sample_eval_cases)

        assert report.total_cases == 5
        assert report.passed == 0
        assert report.failed == 5
        assert report.accuracy == 0.0
        assert report.avg_score == 0.0

    def test_partial_agent(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        report = evaluator.run(partial_agent, sample_eval_cases)

        assert report.total_cases == 5
        assert report.passed == 2  # 2+2 and 10*5
        assert report.accuracy == 0.4

    def test_error_handling(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        report = evaluator.run(error_agent, sample_eval_cases)

        assert report.total_cases == 5
        assert report.errored == 5
        assert report.passed == 0
        assert all(r.error is not None for r in report.results)

    def test_empty_eval_set(self):
        evaluator = BatchEvaluator()
        report = evaluator.run(good_agent, [])

        assert report.total_cases == 0
        assert report.accuracy == 0.0

    def test_pass_threshold(self, sample_eval_cases):
        # With threshold 1.0, only exact matches pass
        evaluator = BatchEvaluator(pass_threshold=1.0)
        report = evaluator.run(good_agent, sample_eval_cases)
        assert report.passed == 5  # All exact matches

        # With threshold 0.9, fuzzy (0.8) would not pass
        evaluator2 = BatchEvaluator(pass_threshold=0.9)
        fuzzy_case = [EvalCase(task_input="test", expected_output="Paris")]

        def fuzzy_agent(t):
            return "The answer is Paris obviously"

        report2 = evaluator2.run(fuzzy_agent, fuzzy_case)
        assert report2.passed == 0  # 0.8 < 0.9

    def test_per_tag_stats(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        report = evaluator.run(good_agent, sample_eval_cases)

        assert "math" in report.tag_stats
        assert report.tag_stats["math"]["total"] == 3
        assert report.tag_stats["math"]["passed"] == 3
        assert report.tag_stats["math"]["accuracy"] == 1.0

        assert "geography" in report.tag_stats
        assert report.tag_stats["geography"]["total"] == 1

    def test_duration_tracked(self, sample_eval_cases):
        evaluator = BatchEvaluator()
        report = evaluator.run(good_agent, sample_eval_cases)
        assert report.duration_seconds > 0
        assert all(r.duration_seconds >= 0 for r in report.results)

    def test_signal_based_evaluation(self):
        """Test evaluation using an OutcomeSignal instead of expected_output."""

        class MockSignal:
            def evaluate(self, trace: Trace) -> Outcome:
                if "correct" in trace.final_output:
                    return Outcome(status=OutcomeStatus.SUCCESS, score=0.9)
                return Outcome(status=OutcomeStatus.FAILURE, score=0.2)

        cases = [
            EvalCase(task_input="q1"),
            EvalCase(task_input="q2"),
        ]

        def agent(task):
            return "correct answer" if task == "q1" else "wrong"

        evaluator = BatchEvaluator(signal=MockSignal(), pass_threshold=0.7)
        report = evaluator.run(agent, cases)

        assert report.total_cases == 2
        assert report.passed == 1
        assert report.results[0].score == 0.9
        assert report.results[1].score == 0.2


# === TestEvalRunStorage ===


class TestEvalRunStorage:
    def test_store_and_get(self, tmp_path):
        store = LocalStore(path=str(tmp_path / "knowledge"))
        report = EvalRunReport(
            name="test_run",
            total_cases=5,
            passed=3,
            failed=2,
            accuracy=0.6,
            avg_score=0.7,
            results=[
                EvalResult(
                    eval_id="e1", task_input="q1", agent_output="a1", score=1.0, passed=True
                ),
                EvalResult(
                    eval_id="e2", task_input="q2", agent_output="a2", score=0.0, passed=False
                ),
            ],
        )
        store.store_eval_run(report)

        retrieved = store.get_eval_run(report.run_id)
        assert retrieved is not None
        assert retrieved.name == "test_run"
        assert retrieved.total_cases == 5
        assert retrieved.passed == 3
        assert len(retrieved.results) == 2
        assert retrieved.results[0].score == 1.0
        store.close()

    def test_list_eval_runs(self, tmp_path):
        store = LocalStore(path=str(tmp_path / "knowledge"))
        for i in range(3):
            report = EvalRunReport(name=f"run_{i}", total_cases=i + 1)
            store.store_eval_run(report)

        runs = store.list_eval_runs(limit=10)
        assert len(runs) == 3
        store.close()

    def test_partial_id_match(self, tmp_path):
        store = LocalStore(path=str(tmp_path / "knowledge"))
        report = EvalRunReport(name="test")
        store.store_eval_run(report)

        retrieved = store.get_eval_run(report.run_id[:8])
        assert retrieved is not None
        assert retrieved.run_id == report.run_id
        store.close()

    def test_get_nonexistent(self, tmp_path):
        store = LocalStore(path=str(tmp_path / "knowledge"))
        assert store.get_eval_run("nonexistent") is None
        store.close()


# === TestEvalComparison ===


class TestEvalComparison:
    def test_compare_improvement(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        baseline = evaluator.run(bad_agent, sample_eval_cases, name="baseline")
        treatment = evaluator.run(good_agent, sample_eval_cases, name="treatment")

        comp = BatchEvaluator.compare(baseline, treatment)
        assert comp.accuracy_delta == 1.0  # 0% → 100%
        assert comp.score_delta == 1.0
        assert len(comp.improvements) == 5
        assert len(comp.regressions) == 0

    def test_compare_regression(self, sample_eval_cases):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        baseline = evaluator.run(good_agent, sample_eval_cases, name="baseline")
        treatment = evaluator.run(bad_agent, sample_eval_cases, name="treatment")

        comp = BatchEvaluator.compare(baseline, treatment)
        assert comp.accuracy_delta == -1.0
        assert len(comp.regressions) == 5
        assert len(comp.improvements) == 0

    def test_compare_partial_overlap(self):
        """Compare runs with different eval sets — only common cases matched."""
        baseline = EvalRunReport(
            name="baseline",
            accuracy=0.5,
            avg_score=0.5,
            results=[
                EvalResult(
                    eval_id="e1", task_input="q1", agent_output="a1", score=1.0, passed=True
                ),
                EvalResult(
                    eval_id="e2", task_input="q2", agent_output="a2", score=0.0, passed=False
                ),
            ],
        )
        treatment = EvalRunReport(
            name="treatment",
            accuracy=1.0,
            avg_score=1.0,
            results=[
                EvalResult(
                    eval_id="e1", task_input="q1", agent_output="a1", score=1.0, passed=True
                ),
                EvalResult(
                    eval_id="e3", task_input="q3", agent_output="a3", score=1.0, passed=True
                ),
            ],
        )
        comp = BatchEvaluator.compare(baseline, treatment)
        # Only q1 is common — no change (both 1.0)
        assert len(comp.improvements) == 0
        assert len(comp.regressions) == 0


# === TestEvalManagerImport ===


def _mock_get_embedding(text):
    """Seeded random embedding for deterministic tests."""
    import numpy as np

    np.random.seed(hash(text) % 2**32)
    vec = np.random.randn(1536).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


class TestEvalManagerImport:
    def test_import_csv_via_engine(self, tmp_path):
        csv_path = tmp_path / "eval.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_input", "expected_output"])
            writer.writerow(["What is 2+2?", "4"])
            writer.writerow(["Capital of France?", "Paris"])

        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))
            count = engine.eval.import_csv(str(csv_path))
            assert count == 2
            assert engine.eval.count() == 2
            engine._store.close()

    def test_import_dicts_via_engine(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))
            data = [
                {"task_input": "q1", "expected_output": "a1", "tags": ["t1"]},
                {"task_input": "q2", "expected_output": "a2"},
            ]
            count = engine.eval.import_dicts(data)
            assert count == 2
            cases = engine.eval.list()
            assert len(cases) == 2
            assert cases[0].source == "dict_import"
            engine._store.close()


# === TestEngineEvaluate ===


class TestEngineEvaluate:
    def test_evaluate_end_to_end(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))

            # Import eval cases
            engine.eval.import_dicts(
                [
                    {"task_input": "What is 2+2?", "expected_output": "4"},
                    {"task_input": "What is 10*5?", "expected_output": "50"},
                    {"task_input": "What is the capital of France?", "expected_output": "Paris"},
                ]
            )

            # Run evaluation
            report = engine.evaluate(good_agent, name="baseline")
            assert report.total_cases == 3
            assert report.accuracy == 1.0
            assert report.name == "baseline"

            # Should be stored
            runs = engine._store.list_eval_runs()
            assert len(runs) == 1
            assert runs[0].run_id == report.run_id

            engine._store.close()

    def test_evaluate_empty_eval_set(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))
            report = engine.evaluate(good_agent)
            assert report.total_cases == 0
            engine._store.close()

    def test_compare_eval_runs(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))

            engine.eval.import_dicts(
                [
                    {"task_input": "What is 2+2?", "expected_output": "4"},
                    {"task_input": "What is 10*5?", "expected_output": "50"},
                ]
            )

            r1 = engine.evaluate(bad_agent, name="before")
            r2 = engine.evaluate(good_agent, name="after")

            comp = engine.compare_eval_runs(r1.run_id, r2.run_id)
            assert comp.accuracy_delta == 1.0
            assert len(comp.improvements) == 2

            engine._store.close()

    def test_compare_nonexistent_run(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))
            with pytest.raises(ValueError, match="Eval run not found"):
                engine.compare_eval_runs("nonexistent", "also_nonexistent")
            engine._store.close()

    def test_evaluate_with_tags_filter(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=_mock_get_embedding):
            from agentlearn import Engine

            engine = Engine(store=str(tmp_path / "knowledge"))

            engine.eval.import_dicts(
                [
                    {"task_input": "What is 2+2?", "expected_output": "4", "tags": ["math"]},
                    {
                        "task_input": "Capital of France?",
                        "expected_output": "Paris",
                        "tags": ["geography"],
                    },
                ]
            )

            report = engine.evaluate(good_agent, tags=["math"])
            assert report.total_cases == 1
            engine._store.close()


class TestAsyncBatchEvaluator:
    """Tests for async batch evaluation."""

    @pytest.mark.asyncio
    async def test_async_run_with_sync_agent(self):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        cases = [
            EvalCase(task_input="What is 2+2?", expected_output="4"),
            EvalCase(task_input="Capital of France?", expected_output="Paris"),
        ]

        def sync_agent(task):
            if "2+2" in task:
                return "4"
            return "Paris"

        report = await evaluator.run_async(sync_agent, cases)
        assert report.total_cases == 2
        assert report.passed == 2
        assert report.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_async_run_with_async_agent(self):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        cases = [
            EvalCase(task_input="hello", expected_output="HELLO"),
        ]

        async def async_agent(task):
            return task.upper()

        report = await evaluator.run_async(async_agent, cases)
        assert report.total_cases == 1
        assert report.passed == 1

    @pytest.mark.asyncio
    async def test_async_run_error_handling(self):
        evaluator = BatchEvaluator(pass_threshold=0.7)
        cases = [EvalCase(task_input="crash")]

        async def failing_agent(task):
            raise RuntimeError("boom")

        report = await evaluator.run_async(failing_agent, cases)
        assert report.errored == 1
        assert report.results[0].error == "boom"

    @pytest.mark.asyncio
    async def test_async_run_empty_cases(self):
        evaluator = BatchEvaluator()
        report = await evaluator.run_async(lambda x: x, [])
        assert report.total_cases == 0
