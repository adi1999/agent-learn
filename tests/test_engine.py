"""Tests for the Engine orchestrator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentlearn.engine import Engine, EvalManager, KnowledgeManager
from agentlearn.models import (
    CandidateFix,
    FixType,
    InjectionResult,
    KnowledgeItem,
    KnowledgeStatus,
    LearningReport,
    Outcome,
    OutcomeStatus,
    Trace,
    ValidationResult,
)


def mock_get_embedding(text, *args, **kwargs):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).astype(np.float32)


@pytest.fixture
def engine(tmp_path):
    """Engine with mocked external dependencies."""
    with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
        e = Engine(
            store=str(tmp_path / "knowledge"),
            model="gpt-4o-mini",
        )
        yield e
        e._store.close()


class TestEngineInit:
    def test_default_components(self, engine):
        assert engine._tracer is not None
        assert engine._analyzer is not None
        assert engine._validator is not None
        assert engine._signal is not None
        assert engine._injector is not None
        assert engine._store is not None

    def test_custom_store_path(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(store=str(tmp_path / "custom_store"))
            assert e._store.path == str(tmp_path / "custom_store")
            e._store.close()

    def test_injection_toggle(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(store=str(tmp_path / "store"), injection_enabled=False)
            assert e.injection_enabled is False
            e._store.close()


class TestTraceDecorator:
    def test_basic_trace(self, engine):
        """Test that the trace decorator runs the function and stores a trace."""
        # Mock the signal to avoid real LLM calls
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(
            status=OutcomeStatus.SUCCESS, score=0.9
        )

        @engine.trace
        def my_agent(task_input):
            return f"Result for: {task_input}"

        result = my_agent("test task")
        assert result == "Result for: test task"

        # Verify trace was stored
        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].task_input == "test task"
        assert traces[0].outcome.status == OutcomeStatus.SUCCESS

    def test_get_knowledge(self, engine):
        """Test that get_knowledge returns relevant knowledge."""
        # Store active knowledge
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Always check for null",
                applies_when="Processing data",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        knowledge = engine.get_knowledge("Process some data")
        # Should return a string (may be empty if embeddings don't match)
        assert isinstance(knowledge, str)

    def test_trace_handles_exception(self, engine):
        """Test that exceptions are recorded and re-raised."""
        engine._signal = MagicMock()

        @engine.trace
        def failing_agent(task_input):
            raise ValueError("Agent crashed")

        with pytest.raises(ValueError, match="Agent crashed"):
            failing_agent("test")

        # Trace should still be stored with failure outcome
        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].outcome.status == OutcomeStatus.FAILURE

    def test_trace_without_injection(self, engine):
        """Test with injection disabled."""
        engine.injection_enabled = False
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(
            status=OutcomeStatus.SUCCESS, score=0.9
        )

        @engine.trace
        def my_agent(task_input):
            # get_knowledge returns "" when injection disabled
            assert engine.get_knowledge(task_input) == ""
            return "done"

        my_agent("test")

    def test_effectiveness_updated(self, engine):
        """Test that effectiveness is updated after traced runs."""
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(
            status=OutcomeStatus.SUCCESS, score=0.9
        )

        # Create active knowledge
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Test knowledge",
                applies_when="Any task",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        # Mock injector to always inject our item
        engine._injector = MagicMock()
        engine._injector.inject.return_value = InjectionResult(
            system_prompt_additions="test knowledge",
            items_injected=[item.item_id],
        )

        @engine.trace
        def my_agent(task_input):
            # Call get_knowledge to trigger injection tracking
            engine.get_knowledge(task_input)
            return "done"

        my_agent("test")

        updated = engine._store.get(item.item_id)
        assert updated.times_injected == 1
        assert updated.times_helped == 1


class TestRunMethod:
    def test_run(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(
            status=OutcomeStatus.SUCCESS, score=0.9
        )

        def my_agent(task_input):
            return f"Result: {task_input}"

        result = engine.run(my_agent, "test")
        assert result == "Result: test"


class TestLearnCycle:
    def test_learn_no_traces(self, engine):
        report = engine.learn()
        assert report.traces_analyzed == 0
        assert report.candidates_generated == 0

    def _seed_eval_cases(self, engine, count=5):
        """Add eval cases to avoid quick-win mode."""
        from agentlearn.models import EvalCase
        for i in range(count):
            engine._store.store_eval_case(EvalCase(task_input=f"Eval task {i}"))

    def test_learn_with_mock_analyzer(self, engine):
        """Test the full learn cycle with mocked components."""
        # Seed eval cases to bypass quick-win mode
        self._seed_eval_cases(engine)

        # Store a failed trace
        trace = Trace(
            task_input="Calculate compound interest",
            final_output="$1050",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.2),
        )
        engine._store.store_trace(trace)

        # Mock analyzer to return a candidate
        candidate = CandidateFix(
            fix_type=FixType.SKILL,
            content="Use compound interest formula",
            applies_when="Interest calculation",
            confidence=0.8,
            reasoning="Used simple interest",
            source_trace_ids=[trace.trace_id],
        )
        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [candidate]

        # Mock validator to auto-approve
        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(
            passed=True, improvement=0.3, confidence=1.0
        )

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.traces_analyzed == 1
        assert report.candidates_generated == 1
        assert report.candidates_promoted == 1

        # Knowledge should be stored
        active = engine._store.list_all(status="active")
        assert len(active) == 1
        assert "compound interest" in active[0].content

    def test_learn_rejects_failed_validation(self, engine):
        # Seed eval cases to bypass quick-win mode
        self._seed_eval_cases(engine)

        trace = Trace(
            task_input="test task",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="Bad fix",
                applies_when="test",
                source_trace_ids=[trace.trace_id],
            )
        ]

        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(passed=False)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.candidates_rejected == 1
        assert report.candidates_promoted == 0

    def test_learn_quick_win_mode(self, engine):
        """Test quick-win mode (empty store, no eval cases)."""
        trace = Trace(
            task_input="test",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="Quick win fix",
                applies_when="test",
                confidence=0.8,
                source_trace_ids=[trace.trace_id],
            )
        ]

        # Mock click.confirm to auto-approve
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding), \
             patch("agentlearn.engine.click.confirm", return_value=True), \
             patch("agentlearn.engine.click.echo"):
            report = engine.learn()

        assert report.candidates_promoted == 1


class TestKnowledgeManager:
    def test_list_and_show(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Test",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        items = engine.knowledge.list(status="active")
        assert len(items) == 1

        shown = engine.knowledge.show(item.item_id)
        assert shown.content == "Test"

    def test_approve(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Candidate",
                applies_when="test",
                status=KnowledgeStatus.CANDIDATE,
            )
            engine._store.store(item)

        result = engine.knowledge.approve(item.item_id)
        assert result.status == KnowledgeStatus.ACTIVE

    def test_reject(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="To reject",
                applies_when="test",
                status=KnowledgeStatus.CANDIDATE,
            )
            engine._store.store(item)

        result = engine.knowledge.reject(item.item_id)
        assert result.status == KnowledgeStatus.ARCHIVED

    def test_audit(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            # Active item never injected
            engine._store.store(KnowledgeItem(
                fix_type=FixType.SKILL, content="never used", applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            ))

        report = engine.knowledge.audit()
        assert report.total_active == 1
        assert len(report.never_injected) == 1

    def test_import_sandbox(self, engine):
        items = [
            KnowledgeItem(
                fix_type=FixType.SKILL,
                content=f"Imported {i}",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            for i in range(2)
        ]

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            count = engine.knowledge.import_items(items, sandbox=True)

        assert count == 2
        # Should be candidates, not active (sandboxed)
        candidates = engine._store.list_all(status="candidate")
        assert len(candidates) == 2


class TestEvalManager:
    def test_add_and_list(self, engine):
        case = engine.eval.add(
            task_input="What is 2+2?",
            expected_output="4",
            tags=["math"],
        )
        assert case.source == "manual"

        cases = engine.eval.list()
        assert len(cases) == 1

    def test_count(self, engine):
        for i in range(3):
            engine.eval.add(task_input=f"Task {i}")
        assert engine.eval.count() == 3


class TestStatus:
    def test_empty_status(self, engine):
        status = engine.status()
        assert status.knowledge_total == 0
        assert status.traces_total == 0
        assert status.eval_cases == 0
        assert len(status.next_milestones) > 0

    def test_status_with_data(self, engine):
        engine._store.store_trace(Trace(
            task_input="s", outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.9),
        ))
        engine._store.store_trace(Trace(
            task_input="f", outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        ))

        status = engine.status()
        assert status.traces_total == 2
        assert status.traces_success == 1
        assert status.traces_failure == 1
