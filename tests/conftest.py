"""Shared fixtures for agentlearn tests."""

import os
import tempfile

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
)
from agentlearn.store.local_store import LocalStore


@pytest.fixture
def tmp_store(tmp_path):
    """A LocalStore backed by a temporary directory."""
    store = LocalStore(path=str(tmp_path / "knowledge"))
    yield store
    store.close()


@pytest.fixture
def sample_step():
    return Step(
        step_type=StepType.LLM_CALL,
        input_context="What is 2+2?",
        decision="Call gpt-4o",
        result="4",
        metadata={"model": "gpt-4o", "input_tokens": 10, "output_tokens": 5},
    )


@pytest.fixture
def sample_outcome():
    return Outcome(
        status=OutcomeStatus.SUCCESS,
        score=0.9,
        signal_source="llm_judge",
        reasoning="Correct answer",
    )


@pytest.fixture
def sample_trace(sample_step, sample_outcome):
    return Trace(
        task_input="What is 2+2?",
        final_output="4",
        outcome=sample_outcome,
        steps=[sample_step],
    )


@pytest.fixture
def failed_trace(sample_step):
    return Trace(
        task_input="Calculate compound interest on $1000 at 5% for 3 years",
        final_output="$1050",
        outcome=Outcome(
            status=OutcomeStatus.FAILURE,
            score=0.2,
            reasoning="Incorrect calculation — used simple interest instead of compound",
        ),
        steps=[sample_step],
    )


@pytest.fixture
def sample_knowledge():
    return KnowledgeItem(
        fix_type=FixType.SKILL,
        content="Always use compound interest formula: A = P(1 + r/n)^(nt)",
        applies_when="Calculating interest over multiple periods",
        status=KnowledgeStatus.ACTIVE,
        tags=["finance", "math"],
    )


@pytest.fixture
def sample_candidate():
    return CandidateFix(
        fix_type=FixType.SKILL,
        content="Before calculating interest, determine if simple or compound is needed.",
        applies_when="Any interest calculation task",
        confidence=0.85,
        reasoning="Agent used simple interest when compound was required",
        source_trace_ids=["trace-123"],
    )


@pytest.fixture
def sample_eval_case():
    return EvalCase(
        task_input="What is the compound interest on $1000 at 5% for 3 years?",
        expected_output="$1157.63",
        source="manual",
    )
