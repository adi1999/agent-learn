"""All data models for the agentlearn framework."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    priority: str = "normal"  # "pinned" | "normal"

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    eval_id: str = Field(default_factory=_new_id)
    task_input: str
    expected_output: Optional[str] = None
    judge_prompt: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source: str = "auto"
    scorer: Optional[Callable[[str, str], float]] = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=_utcnow)


# === Batch Evaluation Models ===


class EvalResult(BaseModel):
    """Result of evaluating a single case in a batch run."""

    eval_id: str
    task_input: str
    expected_output: Optional[str] = None
    agent_output: str = ""
    score: float = 0.0
    passed: bool = False
    tags: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


class EvalRunReport(BaseModel):
    """Aggregate results from a batch evaluation run."""

    run_id: str = Field(default_factory=_new_id)
    name: str = ""
    created_at: datetime = Field(default_factory=_utcnow)

    # Aggregate metrics
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    accuracy: float = 0.0
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    duration_seconds: float = 0.0

    # Per-tag breakdown: {"tag": {"total": N, "passed": N, "accuracy": F, "avg_score": F}}
    tag_stats: dict[str, dict] = Field(default_factory=dict)

    # Individual results
    results: list[EvalResult] = Field(default_factory=list)

    # Config
    pass_threshold: float = 0.7
    signal_source: str = ""


class EvalComparison(BaseModel):
    """Comparison between two eval runs."""

    baseline_run_id: str
    treatment_run_id: str
    baseline_accuracy: float = 0.0
    treatment_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    baseline_avg_score: float = 0.0
    treatment_avg_score: float = 0.0
    score_delta: float = 0.0
    regressions: list[dict] = Field(default_factory=list)
    improvements: list[dict] = Field(default_factory=list)


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


class LiftReport(BaseModel):
    treatment_avg: float = 0.0
    control_avg: float = 0.0
    lift: float = 0.0
    significant: bool = False
    treatment_count: int = 0
    control_count: int = 0
    recommendation: str = ""


class BlameReport(BaseModel):
    trace_id: str
    candidates: list[dict] = Field(default_factory=list)
    recommendation: str = ""
