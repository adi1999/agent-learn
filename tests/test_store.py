"""Tests for LocalStore."""

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from agentlearn.models import (
    EvalCase,
    KnowledgeItem,
    KnowledgeStatus,
    FixType,
    Outcome,
    OutcomeStatus,
    Step,
    StepType,
    Trace,
)
from agentlearn.store.local_store import LocalStore


# Mock embedding to avoid real API calls
MOCK_EMBEDDING = np.random.rand(1536).astype(np.float32)


def mock_get_embedding(text, *args, **kwargs):
    """Deterministic mock embedding based on text hash."""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).astype(np.float32)


@pytest.fixture
def store(tmp_path):
    """LocalStore with mocked embeddings."""
    with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
        s = LocalStore(path=str(tmp_path / "knowledge"))
        yield s
        s.close()


class TestKnowledgeCRUD:
    def test_store_and_get(self, store):
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="Test skill",
            applies_when="When testing",
            status=KnowledgeStatus.ACTIVE,
        )
        item_id = store.store(item)
        assert item_id == item.item_id

        retrieved = store.get(item.item_id)
        assert retrieved is not None
        assert retrieved.content == "Test skill"
        assert retrieved.status == KnowledgeStatus.ACTIVE

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent-id") is None

    def test_list_all(self, store):
        for i in range(3):
            store.store(KnowledgeItem(
                fix_type=FixType.SKILL,
                content=f"Skill {i}",
                applies_when=f"When {i}",
                status=KnowledgeStatus.ACTIVE,
            ))
        store.store(KnowledgeItem(
            fix_type=FixType.CHECKLIST,
            content="Candidate",
            applies_when="When candidate",
            status=KnowledgeStatus.CANDIDATE,
        ))

        all_items = store.list_all()
        assert len(all_items) == 4

        active = store.list_all(status="active")
        assert len(active) == 3

        candidates = store.list_all(status="candidate")
        assert len(candidates) == 1

    def test_deprecate(self, store):
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="To deprecate",
            applies_when="test",
            status=KnowledgeStatus.ACTIVE,
        )
        store.store(item)
        store.deprecate(item.item_id, reason="test")

        retrieved = store.get(item.item_id)
        assert retrieved.status == KnowledgeStatus.DEPRECATED

    def test_update_effectiveness_helped(self, store):
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="Effective skill",
            applies_when="test",
            status=KnowledgeStatus.ACTIVE,
        )
        store.store(item)

        store.update_effectiveness(item.item_id, helped=True)
        retrieved = store.get(item.item_id)
        assert retrieved.times_injected == 1
        assert retrieved.times_helped == 1
        assert retrieved.effectiveness_rate == 1.0

    def test_update_effectiveness_not_helped(self, store):
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="Ineffective skill",
            applies_when="test",
            status=KnowledgeStatus.ACTIVE,
        )
        store.store(item)

        store.update_effectiveness(item.item_id, helped=False)
        retrieved = store.get(item.item_id)
        assert retrieved.times_injected == 1
        assert retrieved.times_helped == 0
        assert retrieved.effectiveness_rate == 0.0

    def test_effectiveness_rate_calculation(self, store):
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="Mixed skill",
            applies_when="test",
            status=KnowledgeStatus.ACTIVE,
        )
        store.store(item)

        # 3 helped, 2 not helped = 60% rate
        for _ in range(3):
            store.update_effectiveness(item.item_id, helped=True)
        for _ in range(2):
            store.update_effectiveness(item.item_id, helped=False)

        retrieved = store.get(item.item_id)
        assert retrieved.times_injected == 5
        assert retrieved.times_helped == 3
        assert abs(retrieved.effectiveness_rate - 0.6) < 0.01

    def test_count_by_status(self, store):
        store.store(KnowledgeItem(
            fix_type=FixType.SKILL, content="a", applies_when="x",
            status=KnowledgeStatus.ACTIVE,
        ))
        store.store(KnowledgeItem(
            fix_type=FixType.SKILL, content="b", applies_when="y",
            status=KnowledgeStatus.ACTIVE,
        ))
        store.store(KnowledgeItem(
            fix_type=FixType.SKILL, content="c", applies_when="z",
            status=KnowledgeStatus.CANDIDATE,
        ))

        counts = store.count_by_status()
        assert counts.get("active") == 2
        assert counts.get("candidate") == 1

    def test_export_and_import(self, store):
        items = [
            KnowledgeItem(
                fix_type=FixType.SKILL, content=f"Skill {i}", applies_when=f"When {i}",
                status=KnowledgeStatus.ACTIVE,
            )
            for i in range(3)
        ]
        for item in items:
            store.store(item)

        exported = store.export_all()
        assert len(exported) == 3

        # Import into fresh store
        store2 = LocalStore(path=str(store.path + "_import"))
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            count = store2.import_items(exported)
        assert count == 3
        assert len(store2.list_all()) == 3
        store2.close()


class TestTraceCRUD:
    def test_store_and_get_trace(self, store):
        trace = Trace(
            task_input="Test task",
            final_output="Test output",
            outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.9),
            steps=[
                Step(
                    step_type=StepType.LLM_CALL,
                    input_context="context",
                    decision="call model",
                    result="output",
                )
            ],
        )
        store.store_trace(trace)

        retrieved = store.get_trace(trace.trace_id)
        assert retrieved is not None
        assert retrieved.task_input == "Test task"
        assert retrieved.outcome.status == OutcomeStatus.SUCCESS
        assert len(retrieved.steps) == 1

    def test_unanalyzed_traces_failure_first(self, store):
        # Store success trace
        store.store_trace(Trace(
            task_input="success",
            outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.9),
        ))
        # Store failure trace
        store.store_trace(Trace(
            task_input="failure",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        ))

        traces = store.get_unanalyzed_traces(prioritize_failures=True)
        assert len(traces) == 2
        # Failures should come first (ASC ordering: failure < success alphabetically)
        assert traces[0].outcome.status == OutcomeStatus.FAILURE

    def test_mark_trace_analyzed(self, store):
        trace = Trace(task_input="test", outcome=Outcome(status=OutcomeStatus.FAILURE))
        store.store_trace(trace)

        unanalyzed = store.get_unanalyzed_traces()
        assert len(unanalyzed) == 1

        store.mark_trace_analyzed(trace.trace_id)

        unanalyzed = store.get_unanalyzed_traces()
        assert len(unanalyzed) == 0

    def test_get_traces_by_status(self, store):
        store.store_trace(Trace(
            task_input="s1", outcome=Outcome(status=OutcomeStatus.SUCCESS),
        ))
        store.store_trace(Trace(
            task_input="f1", outcome=Outcome(status=OutcomeStatus.FAILURE),
        ))
        store.store_trace(Trace(
            task_input="s2", outcome=Outcome(status=OutcomeStatus.SUCCESS),
        ))

        successes = store.get_traces(status="success")
        assert len(successes) == 2

        failures = store.get_traces(status="failure")
        assert len(failures) == 1

    def test_count_traces(self, store):
        store.store_trace(Trace(
            task_input="s", outcome=Outcome(status=OutcomeStatus.SUCCESS),
        ))
        store.store_trace(Trace(
            task_input="f", outcome=Outcome(status=OutcomeStatus.FAILURE),
        ))

        counts = store.count_traces()
        assert counts["total"] == 2
        assert counts.get("success") == 1
        assert counts.get("failure") == 1
        assert counts["unanalyzed"] == 2


class TestEvalCaseCRUD:
    def test_store_and_list(self, store):
        case = EvalCase(
            task_input="Test question",
            expected_output="Expected answer",
            source="manual",
        )
        store.store_eval_case(case)

        cases = store.list_eval_cases()
        assert len(cases) == 1
        assert cases[0].task_input == "Test question"

    def test_count(self, store):
        for i in range(5):
            store.store_eval_case(EvalCase(task_input=f"Task {i}"))
        assert store.count_eval_cases() == 5

    def test_filter_by_tags(self, store):
        store.store_eval_case(EvalCase(task_input="math", tags=["math"]))
        store.store_eval_case(EvalCase(task_input="code", tags=["code"]))
        store.store_eval_case(EvalCase(task_input="both", tags=["math", "code"]))

        math_cases = store.list_eval_cases(tags=["math"])
        assert len(math_cases) == 2

        code_cases = store.list_eval_cases(tags=["code"])
        assert len(code_cases) == 2


class TestQuery:
    def test_query_returns_relevant(self, store):
        """Test that query returns items (exact relevance depends on embeddings)."""
        item = KnowledgeItem(
            fix_type=FixType.SKILL,
            content="Always validate dates",
            applies_when="When processing date inputs",
            status=KnowledgeStatus.ACTIVE,
        )
        store.store(item)

        results = store.query(task_context="handle a date field", status="active")
        assert len(results) == 1
        assert results[0].item_id == item.item_id

    def test_query_empty_store(self, store):
        results = store.query(task_context="anything")
        assert results == []

    def test_query_respects_status(self, store):
        store.store(KnowledgeItem(
            fix_type=FixType.SKILL, content="active", applies_when="test",
            status=KnowledgeStatus.ACTIVE,
        ))
        store.store(KnowledgeItem(
            fix_type=FixType.SKILL, content="deprecated", applies_when="test",
            status=KnowledgeStatus.DEPRECATED,
        ))

        active = store.query(task_context="test", status="active")
        assert len(active) == 1
        assert active[0].content == "active"
