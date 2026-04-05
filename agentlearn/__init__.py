"""agentlearn — recursive learning framework for AI agents."""

from .engine import Engine
from .evaluator import BatchEvaluator
from .models import (
    AuditReport,
    BlameReport,
    CandidateFix,
    EngineStatus,
    EvalCase,
    EvalCaseResult,
    EvalComparison,
    EvalResult,
    EvalRunReport,
    FixType,
    InjectionResult,
    KnowledgeItem,
    KnowledgeStatus,
    LearningReport,
    LiftReport,
    Outcome,
    OutcomeStatus,
    Step,
    StepType,
    Trace,
    ValidationResult,
)

__version__ = "0.5.0"

__all__ = [
    "Engine",
    "BatchEvaluator",
    "AuditReport",
    "BlameReport",
    "CandidateFix",
    "EngineStatus",
    "EvalCase",
    "EvalCaseResult",
    "EvalComparison",
    "EvalResult",
    "EvalRunReport",
    "FixType",
    "InjectionResult",
    "KnowledgeItem",
    "KnowledgeStatus",
    "LearningReport",
    "LiftReport",
    "Outcome",
    "OutcomeStatus",
    "Step",
    "StepType",
    "Trace",
    "ValidationResult",
]
