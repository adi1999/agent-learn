"""Batch pattern analyzer — finds patterns across multiple traces using FTS5 clustering."""

from __future__ import annotations

import json
from typing import Optional

from ..models import CandidateFix, FixType, Trace
from ..utils.cost_tracker import CostTracker, estimate_cost
from ..utils.llm import get_openai_client, parse_json_from_llm
from ..utils.logging import get_logger

logger = get_logger("batch_analyzer")

BATCH_ANALYSIS_PROMPT = """You are analyzing {n} traces of an agent performing similar tasks.
{success_count} succeeded, {failure_count} failed.

Compare the successful traces with the failed ones. Identify:
1. What do successful runs do differently at key decision points?
2. Are there recurring failure patterns?
3. What systemic fixes would address multiple failures at once?

For each pattern found, generate a fix. Prioritize fixes that address
the most failures.

Output JSON:
{{
  "patterns": [
    {{
      "description": "pattern description",
      "affected_trace_ids": ["..."],
      "fix": {{
        "type": "skill|checklist|routing_rule|anti_pattern|code_template",
        "content": "the actual fix text — readable instructions",
        "applies_when": "condition for when to inject this",
        "confidence": 0.0-1.0
      }}
    }}
  ]
}}"""


def _cluster_key(trace: Trace) -> str:
    """Extract a rough cluster key from a trace for grouping similar traces."""
    # Use first 100 chars of task_input as a rough grouping signal
    return trace.task_input[:100].strip().lower()


class BatchPatternAnalyzer:
    """Finds patterns across multiple traces.

    Uses FTS5 search to cluster related traces, then sends each cluster
    to an LLM for cross-trace pattern analysis. Much cheaper than analyzing
    every trace individually.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        cost_tracker: Optional[CostTracker] = None,
        min_cluster_size: int = 3,
        max_traces_per_cluster: int = 20,
    ):
        self.model = model
        self.cost_tracker = cost_tracker
        self.min_cluster_size = min_cluster_size
        self.max_traces_per_cluster = max_traces_per_cluster

    def analyze_single(self, trace: Trace) -> list[CandidateFix]:
        """Single-trace analysis — delegates to LLMAnalyzer pattern."""
        from .llm_analyzer import LLMAnalyzer

        fallback = LLMAnalyzer(model=self.model, cost_tracker=self.cost_tracker)
        return fallback.analyze_single(trace)

    def analyze_batch(self, traces: list[Trace]) -> list[CandidateFix]:
        """Find patterns across multiple traces using clustering + LLM analysis."""
        if len(traces) < self.min_cluster_size:
            # Not enough traces for batch analysis — fall back to single
            all_fixes = []
            for trace in traces:
                all_fixes.extend(self.analyze_single(trace))
            return all_fixes

        # Cluster traces by similarity
        clusters = self._cluster_traces(traces)

        all_candidates = []
        for cluster_name, cluster_traces in clusters.items():
            if len(cluster_traces) < self.min_cluster_size:
                # Too small for batch — analyze individually
                for trace in cluster_traces:
                    all_candidates.extend(self.analyze_single(trace))
                continue

            # Batch analyze this cluster
            candidates = self._analyze_cluster(cluster_traces)
            all_candidates.extend(candidates)

        return all_candidates

    def analyze_with_fts(self, store, query: str) -> list[CandidateFix]:
        """Use FTS5 to find related traces, then batch analyze them.

        This is the recommended entry point when you have a specific
        pattern to investigate (e.g., "timeout", "rate limit", "parsing error").
        """
        traces = store.search_traces(query, limit=self.max_traces_per_cluster)
        if not traces:
            logger.info(f"No traces found for query: {query}")
            return []
        return self.analyze_batch(traces)

    def _cluster_traces(self, traces: list[Trace]) -> dict[str, list[Trace]]:
        """Group traces into clusters of similar tasks."""
        clusters: dict[str, list[Trace]] = {}

        for trace in traces:
            key = _cluster_key(trace)

            # Find existing cluster with similar key
            placed = False
            for existing_key in list(clusters.keys()):
                if self._keys_similar(key, existing_key):
                    clusters[existing_key].append(trace)
                    placed = True
                    break

            if not placed:
                clusters[key] = [trace]

        # Cap cluster sizes
        for key in clusters:
            clusters[key] = clusters[key][: self.max_traces_per_cluster]

        return clusters

    def _keys_similar(self, a: str, b: str) -> bool:
        """Check if two cluster keys are similar enough to merge."""
        # Simple word overlap heuristic
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        smaller = min(len(words_a), len(words_b))
        return overlap / smaller > 0.5 if smaller > 0 else False

    def _analyze_cluster(self, traces: list[Trace]) -> list[CandidateFix]:
        """Send a cluster of traces to LLM for cross-trace pattern analysis."""
        success_count = sum(1 for t in traces if t.outcome and t.outcome.status.value == "success")
        failure_count = len(traces) - success_count

        # Build compact trace summaries
        summaries = []
        for t in traces:
            summary = {
                "trace_id": t.trace_id,
                "task_input": t.task_input[:300],
                "final_output": (t.final_output or "")[:300],
                "outcome_status": t.outcome.status.value if t.outcome else "unknown",
                "outcome_score": t.outcome.score if t.outcome else None,
                "outcome_reasoning": (t.outcome.reasoning or "")[:200] if t.outcome else None,
                "step_count": len(t.steps),
            }
            # Include key decision points from steps
            if t.steps:
                summary["first_step"] = t.steps[0].decision[:100]
                summary["last_step"] = t.steps[-1].decision[:100]
                # Include any error steps
                errors = [s for s in t.steps if s.step_type.value == "error"]
                if errors:
                    summary["errors"] = [e.result[:200] for e in errors[:3]]
            summaries.append(summary)

        prompt = BATCH_ANALYSIS_PROMPT.format(
            n=len(traces),
            success_count=success_count,
            failure_count=failure_count,
        )

        traces_json = json.dumps(summaries, indent=2, default=str)

        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Here are the traces:\n{traces_json}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            self._track_cost(response)

            parsed = parse_json_from_llm(response.choices[0].message.content or "")
            if parsed is None:
                logger.warning("Failed to parse batch analysis output")
                return []

            candidates = []
            for pattern in parsed.get("patterns", []):
                fix_data = pattern.get("fix", {})
                try:
                    fix_type = FixType(fix_data.get("type", "skill"))
                except ValueError:
                    fix_type = FixType.SKILL

                affected_ids = pattern.get("affected_trace_ids", [])
                # Validate affected IDs exist in our cluster
                valid_ids = {t.trace_id for t in traces}
                affected_ids = [tid for tid in affected_ids if tid in valid_ids]
                if not affected_ids:
                    affected_ids = [
                        t.trace_id
                        for t in traces
                        if t.outcome and t.outcome.status.value != "success"
                    ]

                candidate = CandidateFix(
                    fix_type=fix_type,
                    content=fix_data.get("content", ""),
                    applies_when=fix_data.get("applies_when", ""),
                    source_trace_ids=affected_ids,
                    confidence=fix_data.get("confidence", 0.5),
                    reasoning=pattern.get("description", ""),
                )
                if candidate.content:
                    candidates.append(candidate)

            return candidates

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return []

    def _track_cost(self, response) -> None:
        if self.cost_tracker and response.usage:
            cost = estimate_cost(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self.cost_tracker.record("analysis", cost)
