"""Tests for data models."""

from datetime import datetime, timezone

from agentlearn.models import (
    AuditReport,
    CandidateFix,
    EngineStatus,
    EvalCase,
    EvalCaseResult,
    FixType,
    InjectionResult,
    KnowledgeItem,
    KnowledgeStatus,
    LearningReport,
    Outcome,
    OutcomeStatus,
    Step,
    StepType,
    Trace,
    ValidationResult,
)


class TestEnums:
    def test_step_types(self):
        assert StepType.LLM_CALL.value == "llm_call"
        assert StepType.TOOL_USE.value == "tool_use"
        assert StepType.ERROR.value == "error"

    def test_outcome_statuses(self):
        assert OutcomeStatus.SUCCESS.value == "success"
        assert OutcomeStatus.FAILURE.value == "failure"
        assert OutcomeStatus.PARTIAL.value == "partial"

    def test_fix_types(self):
        assert FixType.SKILL.value == "skill"
        assert FixType.CHECKLIST.value == "checklist"
        assert FixType.ANTI_PATTERN.value == "anti_pattern"

    def test_knowledge_statuses(self):
        assert KnowledgeStatus.CANDIDATE.value == "candidate"
        assert KnowledgeStatus.ACTIVE.value == "active"
        assert KnowledgeStatus.DEPRECATED.value == "deprecated"


class TestStep:
    def test_defaults(self):
        step = Step(
            step_type=StepType.LLM_CALL,
            input_context="hello",
            decision="call model",
            result="world",
        )
        assert step.step_id  # auto-generated
        assert step.timestamp  # auto-generated
        assert step.metadata == {}

    def test_serialization(self):
        step = Step(
            step_type=StepType.LLM_CALL,
            input_context="hello",
            decision="call model",
            result="world",
        )
        data = step.model_dump(mode="json")
        restored = Step.model_validate(data)
        assert restored.step_type == StepType.LLM_CALL
        assert restored.input_context == "hello"


class TestOutcome:
    def test_minimal(self):
        outcome = Outcome(status=OutcomeStatus.SUCCESS)
        assert outcome.score is None
        assert outcome.signal_source == "llm_judge"

    def test_full(self):
        outcome = Outcome(
            status=OutcomeStatus.FAILURE,
            score=0.3,
            reasoning="Bad output",
            criteria=["accuracy", "completeness"],
        )
        assert outcome.score == 0.3
        assert len(outcome.criteria) == 2


class TestTrace:
    def test_defaults(self):
        trace = Trace(task_input="test")
        assert trace.trace_id  # auto-generated
        assert trace.agent_id == "default"
        assert trace.steps == []
        assert trace.analyzed is False
        assert trace.cost_usd == 0.0

    def test_with_outcome(self, sample_trace):
        assert sample_trace.outcome.status == OutcomeStatus.SUCCESS
        assert len(sample_trace.steps) == 1

    def test_serialization(self, sample_trace):
        data = sample_trace.model_dump(mode="json")
        restored = Trace.model_validate(data)
        assert restored.trace_id == sample_trace.trace_id
        assert restored.outcome.status == OutcomeStatus.SUCCESS


class TestCandidateFix:
    def test_defaults(self):
        fix = CandidateFix(
            fix_type=FixType.SKILL,
            content="Do X",
            applies_when="When Y",
        )
        assert fix.fix_id  # auto-generated
        assert fix.confidence == 0.0
        assert fix.source_trace_ids == []

    def test_serialization(self, sample_candidate):
        data = sample_candidate.model_dump(mode="json")
        restored = CandidateFix.model_validate(data)
        assert restored.fix_type == FixType.SKILL
        assert restored.confidence == 0.85


class TestKnowledgeItem:
    def test_defaults(self):
        item = KnowledgeItem(
            fix_type=FixType.CHECKLIST,
            content="Check A, B, C",
            applies_when="Before processing",
        )
        assert item.status == KnowledgeStatus.CANDIDATE
        assert item.times_injected == 0
        assert item.effectiveness_rate == 0.0

    def test_serialization(self, sample_knowledge):
        data = sample_knowledge.model_dump(mode="json")
        restored = KnowledgeItem.model_validate(data)
        assert restored.fix_type == FixType.SKILL
        assert restored.status == KnowledgeStatus.ACTIVE
        assert "finance" in restored.tags


class TestValidationModels:
    def test_eval_case(self):
        case = EvalCase(task_input="test task")
        assert case.eval_id  # auto-generated
        assert case.source == "auto"
        assert case.expected_output is None

    def test_validation_result(self):
        result = ValidationResult(passed=True, improvement=0.15, confidence=0.95)
        assert result.passed
        assert result.regression_count == 0

    def test_eval_case_result(self):
        ecr = EvalCaseResult(
            task_input="test",
            baseline_score=0.6,
            treatment_score=0.8,
            passed=True,
        )
        assert ecr.treatment_score > ecr.baseline_score


class TestReportModels:
    def test_learning_report_defaults(self):
        report = LearningReport()
        assert report.traces_analyzed == 0
        assert report.total_cost_usd == 0.0

    def test_engine_status_defaults(self):
        status = EngineStatus()
        assert status.knowledge_active == 0
        assert status.next_milestones == []

    def test_audit_report_defaults(self):
        report = AuditReport()
        assert report.total_active == 0
        assert report.declining_effectiveness == []


class TestInjectionResult:
    def test_defaults(self):
        result = InjectionResult()
        assert result.system_prompt_additions == ""
        assert result.items_injected == []


class TestIdGeneration:
    def test_unique_ids(self):
        t1 = Trace(task_input="a")
        t2 = Trace(task_input="b")
        assert t1.trace_id != t2.trace_id

        k1 = KnowledgeItem(fix_type=FixType.SKILL, content="x", applies_when="y")
        k2 = KnowledgeItem(fix_type=FixType.SKILL, content="x", applies_when="y")
        assert k1.item_id != k2.item_id


class TestTimestamps:
    def test_auto_timestamp(self):
        step = Step(
            step_type=StepType.OUTPUT,
            input_context="",
            decision="",
            result="",
        )
        assert step.timestamp.tzinfo is not None
        assert step.timestamp.tzinfo == timezone.utc
