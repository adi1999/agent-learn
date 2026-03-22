"""LLM-based trace analyzer — extracts lessons from failed traces."""

import json
from typing import Optional

from ..models import CandidateFix, FixType, Trace
from ..utils.cost_tracker import CostTracker, estimate_cost
from ..utils.llm import get_openai_client, parse_json_from_llm
from ..utils.logging import get_logger

logger = get_logger("analyzer")

SINGLE_TRACE_PROMPT = """You are an AI agent failure analyst. You will receive a trace of an agent
execution that failed (or partially failed). Your job is to:

1. Identify which step(s) caused the failure
2. Determine what knowledge the agent was missing
3. Generate a concrete fix — a skill, checklist, or rule — that would prevent
   this failure in future runs

The fix must be:
- Specific and actionable (not vague advice)
- Scoped to when it applies (not "always do X")
- Written as an instruction the agent can follow
- Self-contained (no external references)

Output JSON:
{
  "root_cause": "one sentence",
  "critical_step_id": "step that caused failure",
  "missing_knowledge": "what the agent needed to know",
  "fix": {
    "type": "skill|checklist|routing_rule|anti_pattern|code_template",
    "content": "the actual fix text — readable instructions",
    "applies_when": "condition for when to inject this",
    "confidence": 0.0-1.0
  }
}"""


class LLMAnalyzer:
    """Analyzes failed traces using an LLM to extract candidate fixes."""

    def __init__(
        self,
        model: str = "gpt-4o",
        cost_tracker: Optional[CostTracker] = None,
        max_trace_tokens: int = 8000,
    ):
        self.model = model
        self.cost_tracker = cost_tracker
        self.max_trace_tokens = max_trace_tokens

    def analyze_single(self, trace: Trace) -> list[CandidateFix]:
        """Analyze one trace and return candidate fixes."""
        client = get_openai_client()
        trace_json = self._prepare_trace(trace)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SINGLE_TRACE_PROMPT},
                    {"role": "user", "content": f"Here is the trace:\n{trace_json}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            self._track_cost(response)

            parsed = parse_json_from_llm(response.choices[0].message.content or "")
            if parsed is None:
                logger.warning(f"Failed to parse analyzer output for trace {trace.trace_id}")
                return []

            fix_data = parsed.get("fix", {})
            fix_type_str = fix_data.get("type", "skill")

            # Map string to FixType enum
            try:
                fix_type = FixType(fix_type_str)
            except ValueError:
                fix_type = FixType.SKILL

            candidate = CandidateFix(
                fix_type=fix_type,
                content=fix_data.get("content", ""),
                applies_when=fix_data.get("applies_when", ""),
                source_trace_ids=[trace.trace_id],
                confidence=fix_data.get("confidence", 0.5),
                reasoning=parsed.get("root_cause", ""),
            )

            if candidate.content:
                return [candidate]
            return []

        except Exception as e:
            logger.error(f"Analysis failed for trace {trace.trace_id}: {e}")
            return []

    def analyze_batch(self, traces: list[Trace]) -> list[CandidateFix]:
        """Analyze multiple traces. Phase 1: loops over analyze_single."""
        all_candidates = []
        for trace in traces:
            candidates = self.analyze_single(trace)
            all_candidates.extend(candidates)
        return all_candidates

    def _prepare_trace(self, trace: Trace) -> str:
        """Serialize trace for LLM analysis, truncating if needed."""
        data = {
            "trace_id": trace.trace_id,
            "task_input": trace.task_input,
            "final_output": trace.final_output,
            "outcome": trace.outcome.model_dump(mode="json") if trace.outcome else None,
            "steps": [],
        }

        # Add steps, truncating if too many
        steps = trace.steps
        if len(steps) > 20:
            # Keep first 5, last 5, summarize middle
            kept = steps[:5] + steps[-5:]
            data["steps"] = [s.model_dump(mode="json") for s in kept]
            data["note"] = f"Truncated: showing 10 of {len(steps)} steps (first 5 + last 5)"
        else:
            data["steps"] = [s.model_dump(mode="json") for s in steps]

        result = json.dumps(data, indent=2, default=str)

        # Rough token estimate (4 chars per token)
        if len(result) / 4 > self.max_trace_tokens:
            # Truncate step results
            for step_data in data["steps"]:
                if len(step_data.get("result", "")) > 500:
                    step_data["result"] = step_data["result"][:500] + "... [truncated]"
                if len(step_data.get("input_context", "")) > 500:
                    step_data["input_context"] = step_data["input_context"][:500] + "... [truncated]"
            result = json.dumps(data, indent=2, default=str)

        return result

    def _track_cost(self, response) -> None:
        """Track cost from an API response."""
        if self.cost_tracker and response.usage:
            cost = estimate_cost(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self.cost_tracker.record("analysis", cost)
