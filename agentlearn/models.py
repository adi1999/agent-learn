"""All data models for the agentlearn framework."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# === Enums ===


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_USE = "tool_use"
    ROUTING = "routing"
    OUTPUT = "output"
    ERROR = "error"


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class FixType(str, Enum):
    SKILL = "skill"
    CHECKLIST = "checklist"
    ROUTING_RULE = "routing_rule"
    ANTI_PATTERN = "anti_pattern"
    CODE_TEMPLATE = "code_template"


class KnowledgeStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# === Core Models ===


class Step(BaseModel):
    step_id: str = Field(default_factory=_new_id)
    step_type: StepType
    input_context: str
    decision: str
    result: str
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict = Field(default_factory=dict)


class Outcome(BaseModel):
    status: OutcomeStatus
    score: Optional[float] = None
    signal_source: str = "llm_judge"
    reasoning: Optional[str] = None
    criteria: Optional[list[str]] = None


class Trace(BaseModel):
    trace_id: str = Field(default_factory=_new_id)
    agent_id: str = "default"
    parent_trace_id: Optional[str] = None
    task_input: str
    final_output: Optional[str] = None
    outcome: Optional[Outcome] = None
    steps: list[Step] = Field(default_factory=list)
    injected_knowledge: list[str] = Field(default_factory=list)
    environment: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    cost_usd: float = 0.0
    analyzed: bool = False


class CandidateFix(BaseModel):
    fix_id: str = Field(default_factory=_new_id)
    fix_type: FixType
    content: str
    applies_when: str
    source_trace_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class KnowledgeItem(BaseModel):
    item_id: str = Field(default_factory=_new_id)
    fix_type: FixType
    content: str
    applies_when: str
    tags: list[str] = Field(default_factory=list)

    # Provenance
    source_trace_ids: list[str] = Field(default_factory=list)
    analyzer_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)

    # Validation
    validation_improvement: Optional[float] = None
    validation_regressions: int = 0

    # Lifecycle
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE

    # Effectiveness tracking
    times_injected: int = 0
    times_helped: int = 0
    effectiveness_rate: float = 0.0
    last_injected_at: Optional[datetime] = None
    last_validated_at: Optional[datetime] = None


class InjectionResult(BaseModel):
    system_prompt_additions: str = ""
    items_injected: list[str] = Field(default_factory=list)


# === Validation Models ===


class EvalCaseResult(BaseModel):
    task_input: str
    baseline_score: float
    treatment_score: float
    passed: bool


class ValidationResult(BaseModel):
    passed: bool
    improvement: float = 0.0
    regression_count: int = 0
    confidence: float = 0.0
    details: list[EvalCaseResult] = Field(default_factory=list)
    cost_usd: float = 0.0


class EvalCase(BaseModel):
    eval_id: str = Field(default_factory=_new_id)
    task_input: str
    expected_output: Optional[str] = None
    judge_prompt: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source: str = "auto"
    created_at: datetime = Field(default_factory=_utcnow)


# === Report Models ===


class LearningReport(BaseModel):
    traces_analyzed: int = 0
    candidates_generated: int = 0
    candidates_validated: int = 0
    candidates_promoted: int = 0
    candidates_rejected: int = 0
    knowledge_deprecated: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    details: list[dict] = Field(default_factory=list)


class EngineStatus(BaseModel):
    # Knowledge counts
    knowledge_active: int = 0
    knowledge_candidate: int = 0
    knowledge_deprecated: int = 0
    knowledge_total: int = 0

    # Trace counts
    traces_total: int = 0
    traces_success: int = 0
    traces_failure: int = 0
    traces_partial: int = 0
    traces_unanalyzed: int = 0

    # Eval set
    eval_cases: int = 0

    # Effectiveness
    avg_effectiveness: float = 0.0

    # Progress indicators
    baseline_accuracy: Optional[float] = None
    current_accuracy: Optional[float] = None
    improvement: Optional[float] = None

    # Next milestones
    next_milestones: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    declining_effectiveness: list[dict] = Field(default_factory=list)
    never_injected: list[str] = Field(default_factory=list)
    insufficient_data: list[str] = Field(default_factory=list)
    total_active: int = 0
    avg_effectiveness: float = 0.0
