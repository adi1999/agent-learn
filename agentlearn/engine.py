"""Core engine — ties all components together and provides the public API."""

from __future__ import annotations

import functools
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import click

from .analyzers.llm_analyzer import LLMAnalyzer
from .injector.embedding import EmbeddingInjector
from .models import (
    AuditReport,
    CandidateFix,
    EngineStatus,
    EvalCase,
    InjectionResult,
    KnowledgeItem,
    KnowledgeStatus,
    LearningReport,
    Outcome,
    OutcomeStatus,
    Trace,
)
from .signals.llm_judge import LLMJudge
from .store.local_store import LocalStore
from .tracers.generic_llm import GenericLLMTracer
from .utils.cost_tracker import CostTracker
from .utils.embeddings import cosine_similarity, get_embedding
from .utils.logging import get_logger
from .validators.human_validator import HumanInLoopValidator

logger = get_logger("engine")

# Auto-deprecation thresholds
DEPRECATION_THRESHOLD = 0.3
MIN_INJECTIONS_FOR_JUDGMENT = 10

# Quick-win mode thresholds
QUICK_WIN_EVAL_THRESHOLD = 5


class KnowledgeManager:
    """Manages knowledge: list, show, approve, reject, deprecate, export, import, audit."""

    def __init__(self, store: LocalStore):
        self._store = store

    def list(self, status: Optional[str] = None) -> list[KnowledgeItem]:
        return self._store.list_all(status=status)

    def show(self, item_id: str) -> Optional[KnowledgeItem]:
        return self._store.get(item_id)

    def approve(self, item_id: str) -> Optional[KnowledgeItem]:
        """Promote a candidate/validated item to active."""
        item = self._store.get(item_id)
        if item is None:
            return None
        if item.status not in (KnowledgeStatus.CANDIDATE, KnowledgeStatus.VALIDATED):
            logger.warning(f"Cannot approve item with status {item.status.value}")
            return None
        item.status = KnowledgeStatus.ACTIVE
        self._store.update_item(item)
        return item

    def reject(self, item_id: str, reason: str = "") -> Optional[KnowledgeItem]:
        """Reject a candidate item."""
        item = self._store.get(item_id)
        if item is None:
            return None
        item.status = KnowledgeStatus.ARCHIVED
        self._store.update_item(item)
        return item

    def deprecate(self, item_id: str, reason: str = "") -> None:
        self._store.deprecate(item_id, reason)

    def export(self) -> list[KnowledgeItem]:
        return self._store.export_all()

    def import_items(self, items: list[KnowledgeItem], sandbox: bool = True) -> int:
        """Import items. If sandbox=True, all enter as 'candidate'."""
        if sandbox:
            for item in items:
                item.status = KnowledgeStatus.CANDIDATE
        return self._store.import_items(items)

    def audit(self) -> AuditReport:
        """Analyze knowledge health."""
        active = self._store.list_all(status="active")

        declining = []
        never_injected = []
        insufficient = []

        for item in active:
            if item.times_injected == 0:
                never_injected.append(item.item_id)
            elif item.times_injected < MIN_INJECTIONS_FOR_JUDGMENT:
                insufficient.append(item.item_id)
            elif item.effectiveness_rate < DEPRECATION_THRESHOLD:
                declining.append({
                    "item_id": item.item_id,
                    "effectiveness_rate": item.effectiveness_rate,
                    "times_injected": item.times_injected,
                })

        avg_eff = 0.0
        rated = [i for i in active if i.times_injected >= MIN_INJECTIONS_FOR_JUDGMENT]
        if rated:
            avg_eff = sum(i.effectiveness_rate for i in rated) / len(rated)

        return AuditReport(
            declining_effectiveness=declining,
            never_injected=never_injected,
            insufficient_data=insufficient,
            total_active=len(active),
            avg_effectiveness=avg_eff,
        )


class EvalManager:
    """Manages eval cases: add, list, count, promote from traces."""

    def __init__(self, store: LocalStore):
        self._store = store

    def add(
        self,
        task_input: str,
        expected_output: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> EvalCase:
        case = EvalCase(
            task_input=task_input,
            expected_output=expected_output,
            judge_prompt=judge_prompt,
            tags=tags or [],
            source="manual",
        )
        self._store.store_eval_case(case)
        return case

    def list(self, tags: Optional[list[str]] = None) -> list[EvalCase]:
        return self._store.list_eval_cases(tags=tags)

    def count(self) -> int:
        return self._store.count_eval_cases()

    def promote_from_traces(self, min_confidence: float = 0.85, limit: int = 20) -> int:
        """Auto-promote high-confidence successful traces to eval cases."""
        traces = self._store.get_traces(status="success", limit=100)
        existing = self._store.list_eval_cases()
        existing_inputs = {c.task_input for c in existing}

        promoted = 0
        for trace in traces:
            if promoted >= limit:
                break
            if not trace.outcome or trace.outcome.score is None:
                continue
            if trace.outcome.score < min_confidence:
                continue
            if trace.task_input in existing_inputs:
                continue

            # Dedup by similarity (skip if too similar to existing)
            too_similar = False
            try:
                trace_emb = get_embedding(trace.task_input)
                for ex in existing:
                    ex_emb = get_embedding(ex.task_input)
                    if cosine_similarity(trace_emb, ex_emb) > 0.9:
                        too_similar = True
                        break
            except Exception:
                pass

            if too_similar:
                continue

            case = EvalCase(
                task_input=trace.task_input,
                expected_output=trace.final_output,
                source="auto_promoted",
            )
            self._store.store_eval_case(case)
            existing_inputs.add(trace.task_input)
            promoted += 1

        return promoted


class Engine:
    """Core orchestrator — ties all components together."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        store: str = "./knowledge",
        tracer=None,
        analyzer=None,
        validator=None,
        signal=None,
        injector=None,
        knowledge_store=None,
        require_human_approval: bool = True,
        injection_enabled: bool = True,
        learning_budget_per_day: Optional[float] = None,
    ):
        self.model = model
        self.injection_enabled = injection_enabled
        self.require_human_approval = require_human_approval

        # Cost tracking
        self._cost_tracker = CostTracker(budget_per_day=learning_budget_per_day)

        # Components — instantiate defaults
        self._store: LocalStore = knowledge_store or LocalStore(path=store)
        self._tracer = tracer or GenericLLMTracer()
        self._analyzer = analyzer or LLMAnalyzer(model=model, cost_tracker=self._cost_tracker)
        self._validator = validator or HumanInLoopValidator()
        self._signal = signal or LLMJudge(model=model, cost_tracker=self._cost_tracker)
        self._injector = injector or EmbeddingInjector()

        # Managers
        self._knowledge_manager = KnowledgeManager(self._store)
        self._eval_manager = EvalManager(self._store)

    # === Public API: Runtime ===

    def trace(self, func):
        """Decorator that auto-traces an agent function.

        The decorated function must accept a `knowledge` keyword argument
        (or **kwargs) to receive injected knowledge.
        """
        @functools.wraps(func)
        def wrapper(task_input: str, **kwargs):
            # 1. Inject knowledge
            injection = InjectionResult()
            if self.injection_enabled:
                try:
                    injection = self._injector.inject(task_input, self._store)
                except Exception as e:
                    logger.warning(f"Injection failed: {e}")

            # 2. Start trace
            trace_id = self._tracer.start_trace(task_input, metadata=kwargs)

            # 3. Run agent function
            try:
                result = func(task_input, knowledge=injection.system_prompt_additions, **kwargs)
            except Exception as e:
                from .models import Step, StepType

                error_step = Step(
                    step_type=StepType.ERROR,
                    input_context=task_input,
                    decision="Run agent function",
                    result=str(e),
                )
                self._tracer.record_step(trace_id, error_step)
                outcome = Outcome(
                    status=OutcomeStatus.FAILURE,
                    score=0.0,
                    reasoning=f"Agent raised exception: {e}",
                )
                trace_obj = self._tracer.end_trace(trace_id, outcome, final_output="")
                trace_obj.injected_knowledge = injection.items_injected
                self._store.store_trace(trace_obj)
                raise

            # 4. Evaluate outcome
            temp_outcome = Outcome(status=OutcomeStatus.PARTIAL, score=0.5)
            trace_obj = self._tracer.end_trace(trace_id, temp_outcome, final_output=str(result))

            try:
                outcome = self._signal.evaluate(trace_obj)
                trace_obj.outcome = outcome
            except Exception as e:
                logger.warning(f"Outcome evaluation failed: {e}")
                trace_obj.outcome = Outcome(
                    status=OutcomeStatus.PARTIAL,
                    score=0.5,
                    reasoning=f"Evaluation failed: {e}",
                )

            # 5. Store trace
            trace_obj.injected_knowledge = injection.items_injected
            self._store.store_trace(trace_obj)

            # 6. Update effectiveness
            for item_id in injection.items_injected:
                helped = trace_obj.outcome.status == OutcomeStatus.SUCCESS
                try:
                    self._store.update_effectiveness(item_id, helped=helped)
                except Exception as e:
                    logger.warning(f"Failed to update effectiveness for {item_id}: {e}")

            return result

        return wrapper

    def run(self, agent_func: Callable, task: str, **kwargs) -> str:
        """Run an agent function with automatic tracing and injection."""
        traced = self.trace(agent_func)
        return traced(task, **kwargs)

    # === Public API: Learning ===

    def learn(self, max_fixes: int = 10) -> LearningReport:
        """Run the full learning cycle on accumulated traces."""
        start_time = time.time()
        report = LearningReport()

        # 1. Fetch unanalyzed traces
        traces = self._store.get_unanalyzed_traces(limit=50, prioritize_failures=True)
        if not traces:
            logger.info("No unanalyzed traces found.")
            return report

        # Check if we're in quick-win mode
        active_count = len(self._store.list_all(status="active"))
        eval_count = self._store.count_eval_cases()
        quick_win = active_count == 0 and eval_count < QUICK_WIN_EVAL_THRESHOLD

        # 2. Analyze traces
        all_candidates: list[CandidateFix] = []
        for trace in traces:
            if not self._cost_tracker.can_spend(0.05):
                logger.warning("Budget exceeded. Stopping analysis.")
                break

            # Only analyze failures and partials
            if trace.outcome and trace.outcome.status == OutcomeStatus.SUCCESS:
                self._store.mark_trace_analyzed(trace.trace_id)
                report.traces_analyzed += 1
                continue

            try:
                candidates = self._analyzer.analyze_single(trace)
                all_candidates.extend(candidates)
            except Exception as e:
                logger.error(f"Analysis failed for trace {trace.trace_id}: {e}")

            self._store.mark_trace_analyzed(trace.trace_id)
            report.traces_analyzed += 1

        report.candidates_generated = len(all_candidates)

        # 3. Validate and promote/reject
        eval_set = self._store.list_eval_cases()

        for candidate in all_candidates[:max_fixes]:
            report.candidates_validated += 1

            if quick_win:
                # Quick-win mode: present directly, skip full validation
                self._quick_win_promote(candidate, report)
            else:
                # Normal mode: full validation
                self._validate_and_promote(candidate, eval_set, report)

        # 4. Auto-deprecate underperforming knowledge
        for item in self._store.list_all(status="active"):
            if (
                item.times_injected >= MIN_INJECTIONS_FOR_JUDGMENT
                and item.effectiveness_rate < DEPRECATION_THRESHOLD
            ):
                self._store.deprecate(item.item_id, reason="effectiveness_decay")
                report.knowledge_deprecated += 1

        report.duration_seconds = time.time() - start_time
        report.total_cost_usd = self._cost_tracker.total_cost
        return report

    def _quick_win_promote(self, candidate: CandidateFix, report: LearningReport) -> None:
        """Quick-win mode: show suggestion, let human approve directly."""
        click.echo()
        click.echo(click.style(
            f"  Suggested fix (confidence: {candidate.confidence:.0%}):", bold=True
        ))
        click.echo(f"    Type: {candidate.fix_type.value}")
        click.echo(f"    Applies when: {candidate.applies_when}")
        click.echo(f"    Content: {candidate.content[:300]}")

        approved = click.confirm("    Approve?", default=False)
        if approved:
            item = KnowledgeItem(
                fix_type=candidate.fix_type,
                content=candidate.content,
                applies_when=candidate.applies_when,
                source_trace_ids=candidate.source_trace_ids,
                status=KnowledgeStatus.ACTIVE,
            )
            self._store.store(item)
            report.candidates_promoted += 1
            click.echo(click.style("    Promoted to active knowledge.", fg="green"))
        else:
            report.candidates_rejected += 1

    def _validate_and_promote(
        self,
        candidate: CandidateFix,
        eval_set: list[EvalCase],
        report: LearningReport,
    ) -> None:
        """Normal validation: run through validator, then promote or reject."""

        def agent_runner(task: str, knowledge_ids: list[str]) -> str:
            # Placeholder — actual agent re-running requires user's agent function
            return ""

        try:
            result = self._validator.validate(candidate, eval_set, agent_runner)
        except Exception as e:
            logger.error(f"Validation failed for {candidate.fix_id}: {e}")
            report.candidates_rejected += 1
            return

        if result.passed:
            item = KnowledgeItem(
                fix_type=candidate.fix_type,
                content=candidate.content,
                applies_when=candidate.applies_when,
                source_trace_ids=candidate.source_trace_ids,
                status=KnowledgeStatus.ACTIVE,
                validation_improvement=result.improvement,
                validation_regressions=result.regression_count,
            )
            self._store.store(item)
            report.candidates_promoted += 1
            report.details.append({
                "fix_id": candidate.fix_id,
                "item_id": item.item_id,
                "action": "promoted",
                "improvement": result.improvement,
            })
        else:
            report.candidates_rejected += 1
            report.details.append({
                "fix_id": candidate.fix_id,
                "action": "rejected",
            })

    # === Public API: Knowledge Management ===

    @property
    def knowledge(self) -> KnowledgeManager:
        return self._knowledge_manager

    # === Public API: Eval Set ===

    @property
    def eval(self) -> EvalManager:
        return self._eval_manager

    # === Public API: Status ===

    def status(self) -> EngineStatus:
        """Current engine status with progress indicators."""
        knowledge_counts = self._store.count_by_status()
        trace_counts = self._store.count_traces()

        # Calculate accuracy trend
        all_traces = self._store.get_traces(limit=200)
        baseline_accuracy = None
        current_accuracy = None
        improvement = None

        if len(all_traces) >= 10:
            # First 10 traces as baseline
            first_10 = all_traces[-10:]  # oldest 10 (list is ordered DESC)
            last_10 = all_traces[:10]  # newest 10

            baseline_success = sum(
                1 for t in first_10 if t.outcome and t.outcome.status == OutcomeStatus.SUCCESS
            )
            current_success = sum(
                1 for t in last_10 if t.outcome and t.outcome.status == OutcomeStatus.SUCCESS
            )

            baseline_accuracy = baseline_success / len(first_10)
            current_accuracy = current_success / len(last_10)
            improvement = current_accuracy - baseline_accuracy

        # Next milestones
        milestones = []
        total_traces = trace_counts.get("total", 0)
        eval_count = self._store.count_eval_cases()
        active_knowledge = knowledge_counts.get("active", 0)

        if total_traces < 20:
            milestones.append(f"Collect {20 - total_traces} more traces to start learning")
        if eval_count < 5:
            milestones.append(f"Add {5 - eval_count} more eval cases to enable validation")
        if active_knowledge == 0:
            milestones.append("Run `engine.learn()` to extract your first knowledge")
        if total_traces >= 20 and total_traces < 50:
            milestones.append(
                f"Collect {50 - total_traces} more traces to unlock batch pattern analysis"
            )

        return EngineStatus(
            knowledge_active=knowledge_counts.get("active", 0),
            knowledge_candidate=knowledge_counts.get("candidate", 0),
            knowledge_deprecated=knowledge_counts.get("deprecated", 0),
            knowledge_total=sum(knowledge_counts.values()),
            traces_total=trace_counts.get("total", 0),
            traces_success=trace_counts.get("success", 0),
            traces_failure=trace_counts.get("failure", 0),
            traces_partial=trace_counts.get("partial", 0),
            traces_unanalyzed=trace_counts.get("unanalyzed", 0),
            eval_cases=eval_count,
            avg_effectiveness=self.knowledge.audit().avg_effectiveness,
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            improvement=improvement,
            next_milestones=milestones,
        )
