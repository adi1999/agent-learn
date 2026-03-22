"""Generic LLM tracer — wraps OpenAI/Anthropic SDK calls."""

import contextvars
import functools
import time
from typing import Optional

from ..models import Outcome, Step, StepType, Trace
from ..utils.cost_tracker import estimate_cost
from ..utils.logging import get_logger

logger = get_logger("tracer")

# Thread-safe context var for active trace_id
_active_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_active_trace_id", default=None
)


class GenericLLMTracer:
    """Tracer that wraps OpenAI and Anthropic SDK calls to auto-record steps."""

    def __init__(self):
        self._active_traces: dict[str, Trace] = {}

    def start_trace(self, task_input: str, metadata: dict = {}) -> str:
        """Begin tracing a new agent run."""
        trace = Trace(
            task_input=task_input,
            agent_id=metadata.get("agent_id", "default"),
            parent_trace_id=metadata.get("parent_trace_id"),
            environment=metadata,
        )
        self._active_traces[trace.trace_id] = trace
        _active_trace_id.set(trace.trace_id)
        logger.debug(f"Started trace {trace.trace_id}")
        return trace.trace_id

    def record_step(self, trace_id: str, step: Step) -> None:
        """Record a step within a trace."""
        trace = self._active_traces.get(trace_id)
        if trace is None:
            logger.warning(f"No active trace for {trace_id}")
            return
        trace.steps.append(step)

    def end_trace(self, trace_id: str, outcome: Outcome, final_output: str = "") -> Trace:
        """Finalize a trace."""
        trace = self._active_traces.pop(trace_id, None)
        if trace is None:
            raise ValueError(f"No active trace for {trace_id}")

        trace.outcome = outcome
        trace.final_output = final_output
        trace.cost_usd = sum(
            s.metadata.get("cost_usd", 0.0) for s in trace.steps
        )

        # Clear context var if it matches
        if _active_trace_id.get() == trace_id:
            _active_trace_id.set(None)

        logger.debug(f"Ended trace {trace_id} with status={outcome.status.value}")
        return trace

    def get_active_trace_id(self) -> Optional[str]:
        """Get the current active trace_id for this context."""
        return _active_trace_id.get()

    def wrap_openai(self, client):
        """Wrap an OpenAI client to auto-record LLM call steps.

        Returns the wrapped client. The original client is modified in place.
        """
        tracer = self
        original_create = client.chat.completions.create

        @functools.wraps(original_create)
        def traced_create(*args, **kwargs):
            trace_id = _active_trace_id.get()
            if trace_id is None:
                return original_create(*args, **kwargs)

            # Check for streaming
            if kwargs.get("stream", False):
                logger.warning("Streaming not supported for tracing in Phase 1. Recording skipped.")
                return original_create(*args, **kwargs)

            start_time = time.time()
            messages = kwargs.get("messages", args[0] if args else [])
            model = kwargs.get("model", "unknown")

            try:
                response = original_create(*args, **kwargs)
            except Exception as e:
                step = Step(
                    step_type=StepType.ERROR,
                    input_context=str(messages[-1] if messages else ""),
                    decision=f"Call {model}",
                    result=str(e),
                    metadata={"model": model, "error": str(e)},
                )
                tracer.record_step(trace_id, step)
                raise

            latency = time.time() - start_time
            usage = response.usage
            cost = estimate_cost(
                model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )

            content = response.choices[0].message.content if response.choices else ""
            step = Step(
                step_type=StepType.LLM_CALL,
                input_context=str(messages[-1] if messages else ""),
                decision=f"Call {model}",
                result=content or "",
                metadata={
                    "model": model,
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "latency_seconds": round(latency, 3),
                    "cost_usd": cost,
                },
            )
            tracer.record_step(trace_id, step)
            return response

        client.chat.completions.create = traced_create
        return client

    def wrap_anthropic(self, client):
        """Wrap an Anthropic client to auto-record LLM call steps.

        Returns the wrapped client. The original client is modified in place.
        """
        tracer = self
        original_create = client.messages.create

        @functools.wraps(original_create)
        def traced_create(*args, **kwargs):
            trace_id = _active_trace_id.get()
            if trace_id is None:
                return original_create(*args, **kwargs)

            if kwargs.get("stream", False):
                logger.warning("Streaming not supported for tracing in Phase 1. Recording skipped.")
                return original_create(*args, **kwargs)

            start_time = time.time()
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")

            try:
                response = original_create(*args, **kwargs)
            except Exception as e:
                step = Step(
                    step_type=StepType.ERROR,
                    input_context=str(messages[-1] if messages else ""),
                    decision=f"Call {model}",
                    result=str(e),
                    metadata={"model": model, "error": str(e)},
                )
                tracer.record_step(trace_id, step)
                raise

            latency = time.time() - start_time
            usage = response.usage
            cost = estimate_cost(
                model,
                usage.input_tokens if usage else 0,
                usage.output_tokens if usage else 0,
            )

            content = ""
            if response.content:
                content = response.content[0].text if response.content[0].type == "text" else str(response.content[0])

            step = Step(
                step_type=StepType.LLM_CALL,
                input_context=str(messages[-1] if messages else ""),
                decision=f"Call {model}",
                result=content,
                metadata={
                    "model": model,
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "latency_seconds": round(latency, 3),
                    "cost_usd": cost,
                },
            )
            tracer.record_step(trace_id, step)
            return response

        client.messages.create = traced_create
        return client
