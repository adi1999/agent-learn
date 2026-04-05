"""Tests for Phase 2 features."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentlearn.models import (
    CandidateFix,
    EvalCase,
    FixType,
    KnowledgeItem,
    KnowledgeStatus,
    Outcome,
    OutcomeStatus,
    Step,
    StepType,
    Trace,
    ValidationResult,
)
from agentlearn.signals.deterministic import (
    CodeExecutionCheck,
    EmptyOutputCheck,
    ErrorStepCheck,
    JSONValidationCheck,
)
from agentlearn.signals.composite import CompositeSignal
from agentlearn.validators.statistical import StatisticalValidator, _welchs_t_test, _mean
from agentlearn.analyzers.batch_analyzer import BatchPatternAnalyzer
from agentlearn.injector.hybrid import HybridInjector, format_knowledge_index
from agentlearn.store.local_store import LocalStore


def mock_get_embedding(text, *args, **kwargs):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).astype(np.float32)


@pytest.fixture
def store(tmp_path):
    with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
        s = LocalStore(path=str(tmp_path / "knowledge"))
        yield s
        s.close()


# === FTS5 Search ===


class TestFTS5Search:
    def test_search_finds_matching_traces(self, store):
        store.store_trace(
            Trace(
                task_input="Calculate compound interest on $1000",
                final_output="$1157.63",
                outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.9),
            )
        )
        store.store_trace(
            Trace(
                task_input="Convert USD to EUR",
                final_output="85 EUR",
                outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.8),
            )
        )

        results = store.search_traces("compound interest")
        assert len(results) == 1
        assert "compound" in results[0].task_input.lower()

    def test_search_no_results(self, store):
        store.store_trace(
            Trace(
                task_input="Hello world",
                outcome=Outcome(status=OutcomeStatus.SUCCESS),
            )
        )
        results = store.search_traces("nonexistent query xyz")
        assert len(results) == 0

    def test_search_multiple_matches(self, store):
        for i in range(5):
            store.store_trace(
                Trace(
                    task_input=f"Parse JSON payload variant {i}",
                    outcome=Outcome(
                        status=OutcomeStatus.FAILURE, score=0.2, reasoning="Invalid JSON format"
                    ),
                )
            )
        store.store_trace(
            Trace(
                task_input="Unrelated task",
                outcome=Outcome(status=OutcomeStatus.SUCCESS),
            )
        )

        results = store.search_traces("JSON")
        assert len(results) == 5


# === Deterministic Checks ===


class TestDeterministicChecks:
    def test_empty_output_fails(self):
        check = EmptyOutputCheck()
        trace = Trace(task_input="test", final_output="")
        result = check.run(trace)
        assert not result.passed
        assert result.hard_fail

    def test_empty_output_passes(self):
        check = EmptyOutputCheck()
        trace = Trace(task_input="test", final_output="Some output")
        result = check.run(trace)
        assert result.passed

    def test_json_valid(self):
        check = JSONValidationCheck()
        trace = Trace(task_input="test", final_output='{"key": "value"}')
        result = check.run(trace)
        assert result.passed

    def test_json_invalid(self):
        check = JSONValidationCheck()
        trace = Trace(task_input="test", final_output='{"key": missing_quotes}')
        result = check.run(trace)
        assert not result.passed
        assert result.hard_fail

    def test_json_in_code_block(self):
        check = JSONValidationCheck()
        trace = Trace(
            task_input="test",
            final_output='Here is the JSON:\n```json\n{"valid": true}\n```',
        )
        result = check.run(trace)
        assert result.passed

    def test_non_json_passes(self):
        check = JSONValidationCheck()
        trace = Trace(task_input="test", final_output="Just a plain text response")
        result = check.run(trace)
        assert result.passed

    def test_error_step_detected(self):
        check = ErrorStepCheck()
        trace = Trace(
            task_input="test",
            steps=[
                Step(step_type=StepType.LLM_CALL, input_context="", decision="", result="ok"),
                Step(
                    step_type=StepType.ERROR,
                    input_context="",
                    decision="",
                    result="Connection timeout",
                ),
            ],
        )
        result = check.run(trace)
        assert not result.passed
        assert not result.hard_fail  # Soft signal

    def test_no_error_steps(self):
        check = ErrorStepCheck()
        trace = Trace(
            task_input="test",
            steps=[
                Step(step_type=StepType.LLM_CALL, input_context="", decision="", result="ok"),
            ],
        )
        result = check.run(trace)
        assert result.passed

    def test_code_syntax_check(self):
        check = CodeExecutionCheck()
        trace = Trace(
            task_input="test",
            final_output="```python\ndef foo():\n    return 42\n```",
        )
        result = check.run(trace)
        assert result.passed

    def test_code_syntax_error(self):
        check = CodeExecutionCheck()
        trace = Trace(
            task_input="test",
            final_output="```python\ndef foo(\n    return 42\n```",
        )
        result = check.run(trace)
        assert not result.passed


# === Composite Signal ===


class TestCompositeSignal:
    def test_hard_fail_deterministic(self):
        signal = CompositeSignal(llm_judge_model="gpt-4o-mini")
        trace = Trace(task_input="test", final_output="")

        outcome = signal.evaluate(trace)
        assert outcome.status == OutcomeStatus.FAILURE
        assert outcome.signal_source == "deterministic"

    def test_delegates_to_llm_on_pass(self):
        """When deterministic passes, LLM judge runs."""
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = Outcome(
            status=OutcomeStatus.SUCCESS, score=0.9, signal_source="llm_judge"
        )

        signal = CompositeSignal()
        signal._llm_judge = mock_judge

        trace = Trace(task_input="test", final_output="Valid output")
        outcome = signal.evaluate(trace)
        assert outcome.status == OutcomeStatus.SUCCESS
        mock_judge.evaluate.assert_called_once()

    def test_add_custom_check(self):
        signal = CompositeSignal()
        initial_count = len(signal.checks)
        signal.add_check(CodeExecutionCheck())
        assert len(signal.checks) == initial_count + 1


# === Statistical Validator ===


class TestStatisticalValidator:
    def test_too_few_eval_cases_defers(self):
        validator = StatisticalValidator(min_eval_cases=5)
        candidate = CandidateFix(fix_type=FixType.SKILL, content="test", applies_when="test")
        eval_set = [EvalCase(task_input=f"t{i}") for i in range(3)]

        # With callback
        result = validator.validate(
            candidate,
            eval_set,
            lambda t, k: "output",
            approval_callback=lambda c: True,
        )
        assert result.passed

    def test_too_few_no_callback_rejects(self):
        validator = StatisticalValidator(min_eval_cases=5)
        candidate = CandidateFix(fix_type=FixType.SKILL, content="test", applies_when="test")
        eval_set = [EvalCase(task_input="t1")]

        result = validator.validate(candidate, eval_set, lambda t, k: "output")
        assert not result.passed

    def test_clear_improvement_passes(self):
        validator = StatisticalValidator(
            noise_margin=0.05,
            max_regressions=0,
            min_confidence=0.5,
            min_eval_cases=3,
        )
        candidate = CandidateFix(fix_type=FixType.SKILL, content="test", applies_when="test")
        eval_set = [
            EvalCase(task_input=f"task {i}", expected_output=f"answer {i}") for i in range(5)
        ]

        # Agent runner: baseline always wrong, treatment always right
        def runner(task, knowledge_ids):
            if knowledge_ids:
                # With fix — return matching answer
                idx = task.split()[-1]
                return f"answer {idx}"
            return "wrong answer"

        result = validator.validate(candidate, eval_set, runner)
        assert result.passed
        assert result.improvement > 0
        assert result.regression_count == 0

    def test_regression_fails(self):
        validator = StatisticalValidator(
            noise_margin=0.0,
            max_regressions=0,
            min_eval_cases=3,
        )
        candidate = CandidateFix(fix_type=FixType.SKILL, content="test", applies_when="test")
        eval_set = [
            EvalCase(task_input="t1", expected_output="a1"),
            EvalCase(task_input="t2", expected_output="a2"),
            EvalCase(task_input="t3", expected_output="a3"),
        ]

        def runner(task, knowledge_ids):
            if knowledge_ids:
                # Fix helps t1 and t3 but REGRESSES t2
                if task == "t1":
                    return "a1"
                elif task == "t2":
                    return "wrong"  # Regression: baseline had this right
                return "a3"
            # Baseline: t2 is correct, t1 and t3 are wrong
            if task == "t2":
                return "a2"
            return "wrong"

        result = validator.validate(candidate, eval_set, runner)
        assert result.regression_count >= 1
        assert not result.passed


class TestStatisticalHelpers:
    def test_mean(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0
        assert _mean([]) == 0.0

    def test_welchs_t_test_identical(self):
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        confidence = _welchs_t_test(a, b)
        assert confidence < 0.6  # No significant difference

    def test_welchs_t_test_clear_difference(self):
        a = [0.3, 0.3, 0.3, 0.3, 0.3]
        b = [0.9, 0.9, 0.9, 0.9, 0.9]
        confidence = _welchs_t_test(a, b)
        assert confidence > 0.9  # Very significant


# === Batch Pattern Analyzer ===


class TestBatchPatternAnalyzer:
    def test_small_batch_falls_back_to_single(self):
        analyzer = BatchPatternAnalyzer(min_cluster_size=5)
        mock_single = MagicMock(return_value=[])

        with patch.object(analyzer, "analyze_single", mock_single):
            traces = [Trace(task_input="test") for _ in range(3)]
            analyzer.analyze_batch(traces)

        assert mock_single.call_count == 3

    def test_clustering(self):
        analyzer = BatchPatternAnalyzer(min_cluster_size=2)
        traces = [
            Trace(task_input="Calculate compound interest on $1000"),
            Trace(task_input="Calculate compound interest on $5000"),
            Trace(task_input="Calculate compound interest on $2000"),
            Trace(task_input="Convert 100 USD to EUR"),
            Trace(task_input="Convert 200 USD to GBP"),
        ]
        clusters = analyzer._cluster_traces(traces)
        # Should have at least 2 clusters (interest vs convert)
        assert len(clusters) >= 2

    def test_fts_search_integration(self, store):
        analyzer = BatchPatternAnalyzer(min_cluster_size=2)

        # Store some traces
        for i in range(5):
            store.store_trace(
                Trace(
                    task_input=f"Parse JSON payload {i}",
                    outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.2),
                )
            )

        # Mock the LLM analysis
        with patch.object(analyzer, "_analyze_cluster", return_value=[]):
            results = analyzer.analyze_with_fts(store, "JSON")
            # Should find traces and attempt batch analysis
            assert isinstance(results, list)


# === Hybrid Injector ===


class TestHybridInjector:
    def test_format_knowledge_index(self):
        items = [
            KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Full content here",
                applies_when="When processing dates",
                status=KnowledgeStatus.ACTIVE,
            ),
        ]
        index = format_knowledge_index(items)
        assert "AVAILABLE KNOWLEDGE" in index
        assert "When processing dates" in index
        # Should NOT contain full content
        assert "Full content here" not in index

    def test_inject_returns_index(self, store):
        injector = HybridInjector(top_k=5)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Detailed instructions here",
                applies_when="When processing dates",
                status=KnowledgeStatus.ACTIVE,
            )
            store.store(item)

        result = injector.inject("process a date field", store)
        # Should contain the index, not the full content
        assert result.items_injected
        assert "AVAILABLE KNOWLEDGE" in result.system_prompt_additions

    def test_recall_full_content(self, store):
        injector = HybridInjector()

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Detailed step-by-step instructions",
                applies_when="When processing dates",
                status=KnowledgeStatus.ACTIVE,
            )
            store.store(item)

        # First inject to populate cache
        injector.inject("date task", store)

        # Then recall by partial ID
        full = injector.recall(item.item_id[:8], store)
        assert "Detailed step-by-step instructions" in full

    def test_recall_not_found(self):
        injector = HybridInjector()
        result = injector.recall("nonexistent")
        assert "not found" in result

    def test_tool_schema(self):
        injector = HybridInjector()
        schema = injector.get_tool_schema()
        assert schema["function"]["name"] == "recall_knowledge"


# === Callback Validator ===


class TestCallbackValidator:
    def test_human_validator_uses_callback(self):
        from agentlearn.validators.human_validator import HumanInLoopValidator

        validator = HumanInLoopValidator()
        candidate = CandidateFix(fix_type=FixType.SKILL, content="test", applies_when="test")

        # With callback — should NOT call click
        result = validator.validate(
            candidate,
            [],
            lambda t, k: "",
            approval_callback=lambda c: True,
        )
        assert result.passed

    def test_human_validator_rejects_via_callback(self):
        from agentlearn.validators.human_validator import HumanInLoopValidator

        validator = HumanInLoopValidator()
        candidate = CandidateFix(fix_type=FixType.SKILL, content="bad fix", applies_when="test")

        result = validator.validate(
            candidate,
            [],
            lambda t, k: "",
            approval_callback=lambda c: False,
        )
        assert not result.passed


# === Engine Phase 2 Integration ===


class TestEnginePhase2:
    @pytest.fixture
    def engine(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            from agentlearn.engine import Engine

            e = Engine(store=str(tmp_path / "knowledge"))
            yield e
            e._store.close()

    def test_get_knowledge_returns_string(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            from agentlearn.engine import Engine

            e = Engine(store=str(tmp_path / "knowledge2"))
            e._signal = MagicMock()
            e._signal.evaluate.return_value = Outcome(
                status=OutcomeStatus.SUCCESS,
                score=0.9,
            )

            @e.trace
            def agent(task_input):
                knowledge = e.get_knowledge(task_input)
                assert isinstance(knowledge, str)
                return "done"

            agent("test")
            e._store.close()

    def test_approval_callback_passed_to_validator(self, engine):
        callback = MagicMock(return_value=True)
        engine._approval_callback = callback

        # Store a trace and mock analyzer
        engine._store.store_trace(
            Trace(
                task_input="test",
                outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
            )
        )
        # Add eval cases to avoid quick-win mode
        for i in range(5):
            engine._store.store_eval_case(EvalCase(task_input=f"eval {i}"))

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="fix",
                applies_when="test",
                source_trace_ids=["t1"],
            )
        ]
        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(passed=True)
        engine.set_agent_runner(lambda task, ids: "dummy")

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            engine.learn()

        # Verify callback was passed through
        engine._validator.validate.assert_called_once()
        call_kwargs = engine._validator.validate.call_args
        assert call_kwargs.kwargs.get("approval_callback") == callback
