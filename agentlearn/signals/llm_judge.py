"""LLM-as-Judge outcome signal — two-step proposer + judge evaluation."""

from typing import Optional

from ..models import Outcome, OutcomeStatus, Trace
from ..utils.cost_tracker import CostTracker, estimate_cost
from ..utils.llm import get_openai_client, parse_json_from_llm
from ..utils.logging import get_logger

logger = get_logger("llm_judge")

PROPOSER_PROMPT = """You are an evaluation criteria generator. Given a task and an agent's output,
generate 3-5 specific, measurable criteria that determine whether the output is good.

Each criterion should be:
- Specific to this task (not generic)
- Binary or clearly scorable
- Focused on correctness, completeness, and usefulness

Output JSON:
{
  "criteria": [
    {"name": "...", "description": "...", "weight": 0.0-1.0},
    ...
  ]
}"""

JUDGE_PROMPT = """You are an output quality judge. Given a task, the agent's output, and
evaluation criteria, score the output on each criterion from 0.0 to 1.0.

Be strict. Only give 1.0 if the criterion is fully satisfied.

Output JSON:
{
  "scores": [
    {"criterion": "...", "score": 0.0-1.0, "reasoning": "..."},
    ...
  ],
  "overall_score": 0.0-1.0,
  "overall_reasoning": "..."
}"""


class LLMJudge:
    """Two-step LLM evaluation: proposer generates criteria, judge scores."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        success_threshold: float = 0.8,
        partial_threshold: float = 0.4,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.model = model
        self.success_threshold = success_threshold
        self.partial_threshold = partial_threshold
        self.cost_tracker = cost_tracker

    def evaluate(self, trace: Trace) -> Outcome:
        """Evaluate a trace using two-step LLM judging."""
        if not trace.final_output:
            return Outcome(
                status=OutcomeStatus.FAILURE,
                score=0.0,
                signal_source="llm_judge",
                reasoning="No output produced",
            )

        client = get_openai_client()

        # Step 1: Propose criteria
        criteria = self._propose_criteria(client, trace.task_input, trace.final_output)
        if criteria is None:
            return Outcome(
                status=OutcomeStatus.PARTIAL,
                score=0.5,
                signal_source="llm_judge",
                reasoning="Failed to generate evaluation criteria",
            )

        # Step 2: Judge against criteria
        result = self._judge(client, trace.task_input, trace.final_output, criteria)
        if result is None:
            return Outcome(
                status=OutcomeStatus.PARTIAL,
                score=0.5,
                signal_source="llm_judge",
                reasoning="Failed to judge output",
            )

        score = result.get("overall_score", 0.5)
        reasoning = result.get("overall_reasoning", "")
        criteria_names = [c.get("name", "") for c in criteria]

        if score >= self.success_threshold:
            status = OutcomeStatus.SUCCESS
        elif score >= self.partial_threshold:
            status = OutcomeStatus.PARTIAL
        else:
            status = OutcomeStatus.FAILURE

        return Outcome(
            status=status,
            score=score,
            signal_source="llm_judge",
            reasoning=reasoning,
            criteria=criteria_names,
        )

    def _propose_criteria(self, client, task_input: str, output: str) -> Optional[list[dict]]:
        """Step 1: Generate evaluation criteria."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROPOSER_PROMPT},
                    {
                        "role": "user",
                        "content": f"Task: {task_input}\n\nAgent Output:\n{output}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            self._track_cost(response)

            parsed = parse_json_from_llm(response.choices[0].message.content or "")
            if parsed and "criteria" in parsed:
                return parsed["criteria"]

            logger.warning("Proposer returned invalid criteria format")
            return None

        except Exception as e:
            logger.error(f"Proposer failed: {e}")
            return None

    def _judge(
        self, client, task_input: str, output: str, criteria: list[dict]
    ) -> Optional[dict]:
        """Step 2: Score output against criteria."""
        import json

        criteria_text = json.dumps(criteria, indent=2)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task_input}\n\n"
                            f"Agent Output:\n{output}\n\n"
                            f"Evaluation Criteria:\n{criteria_text}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            self._track_cost(response)

            parsed = parse_json_from_llm(response.choices[0].message.content or "")
            if parsed and "overall_score" in parsed:
                return parsed

            logger.warning("Judge returned invalid score format")
            return None

        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return None

    def _track_cost(self, response) -> None:
        """Track cost from an API response."""
        if self.cost_tracker and response.usage:
            cost = estimate_cost(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self.cost_tracker.record("outcome_signal", cost)
