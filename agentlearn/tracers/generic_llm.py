"""Generic LLM tracer — wraps OpenAI/Anthropic SDK calls."""

import contextvars
import functools
import time
from typing import Optional

from ..models import Outcome, Step, StepType, Trace
from ..utils.cost_tracker import estimate_cost
from ..utils.logging import get_logger

logger = get_logger("tracer")

# Thread-safe context vars
_active_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_active_trace_id", default=None
)

# Pending knowledge to auto-inject into LLM calls (set by Engine)
_pending_knowledge: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_pending_knowledge", default=""
)


def _inject_knowledge_into_messages(messages: list[dict], knowledge: str) -> list[dict]:
    """Prepend knowledge to the system message in an OpenAI-format message list."""
    if not knowledge:
        return messages

    messages = [dict(m) for m in messages]  # Shallow copy

    # Find and augment the system message
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i] = {**msg, "content": f"{msg['content']}\n\n{knowledge}"}
            return messages

    # No system message — prepend one
    messages.insert(0, {"role": "system", "content": knowledge})
    return messages


class _OpenAIStreamWrapper:
    """Wraps an OpenAI streaming response to buffer content while yielding chunks."""

    def __init__(self, stream, tracer, trace_id: str, model: str, messages, start_time: float):
        self._stream = stream
        self._tracer = tracer
        self._trace_id = trace_id
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._content_parts: list[str] = []
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._record_step()
            raise

        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                self._content_parts.append(delta.content)

        if chunk.usage:
            self._prompt_tokens = chunk.usage.prompt_tokens or 0
            self._completion_tokens = chunk.usage.completion_tokens or 0

        return chunk

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._record_step()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def _record_step(self):
        content = "".join(self._content_parts)
        if not content and not self._content_parts:
            return  # Nothing to record
        latency = time.time() - self._start_time
        cost = estimate_cost(self._model, self._prompt_tokens, self._completion_tokens)
        step = Step(
            step_type=StepType.LLM_CALL,
            input_context=str(self._messages[-1] if self._messages else ""),
            decision=f"Call {self._model} (streamed)",
            result=content,
            metadata={
                "model": self._model,
                "input_tokens": self._prompt_tokens,
                "output_tokens": self._completion_tokens,
                "latency_seconds": round(latency, 3),
                "cost_usd": cost,
                "streamed": True,
            },
        )
        self._tracer.record_step(self._trace_id, step)
        self._content_parts = []  # Prevent double-recording


class _AnthropicStreamWrapper:
    """Wraps an Anthropic streaming response to buffer content while yielding events."""

    def __init__(self, stream, tracer, trace_id: str, model: str, messages, start_time: float):
        self._stream = stream
        self._tracer = tracer
        self._trace_id = trace_id
        self._model = model
        self._messages = messages
        self._start_time = start_time
        self._content_parts: list[str] = []
        self._input_tokens = 0
        self._output_tokens = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
        except StopIteration:
            self._record_step()
            raise

        event_type = getattr(event, "type", "")
        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "text", None):
                self._content_parts.append(delta.text)
        elif event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg and hasattr(msg, "usage"):
                self._input_tokens = getattr(msg.usage, "input_tokens", 0)
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                self._output_tokens = getattr(usage, "output_tokens", 0)

        return event

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._record_step()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return False

    def _record_step(self):
        content = "".join(self._content_parts)
        if not content and not self._content_parts:
            return
        latency = time.time() - self._start_time
        cost = estimate_cost(self._model, self._input_tokens, self._output_tokens)
        step = Step(
            step_type=StepType.LLM_CALL,
            input_context=str(self._messages[-1] if self._messages else ""),
            decision=f"Call {self._model} (streamed)",
            result=content,
            metadata={
                "model": self._model,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "latency_seconds": round(latency, 3),
                "cost_usd": cost,
                "streamed": True,
            },
        )
        self._tracer.record_step(self._trace_id, step)
        self._content_parts = []


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
        trace.cost_usd = sum(s.metadata.get("cost_usd", 0.0) for s in trace.steps)

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

            # Auto-inject: prepend knowledge to system message if available
            knowledge = _pending_knowledge.get("")
            if knowledge:
                kwargs = dict(kwargs)
                messages = list(kwargs.get("messages", args[0] if args else []))
                messages = _inject_knowledge_into_messages(messages, knowledge)
                kwargs["messages"] = messages
                if args:
                    args = (messages, *args[1:])

            messages = kwargs.get("messages", args[0] if args else [])
            model = kwargs.get("model", "unknown")
            start_time = time.time()

            # Streaming: buffer content while passing chunks through
            if kwargs.get("stream", False):
                stream = original_create(*args, **kwargs)
                return _OpenAIStreamWrapper(stream, tracer, trace_id, model, messages, start_time)

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

            # Auto-inject: prepend knowledge to system param for Anthropic
            knowledge = _pending_knowledge.get("")
            if knowledge:
                kwargs = dict(kwargs)
                system = kwargs.get("system", "")
                if system:
                    kwargs["system"] = f"{system}\n\n{knowledge}"
                else:
                    kwargs["system"] = knowledge

            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")
            start_time = time.time()

            # Streaming: buffer content while passing events through
            if kwargs.get("stream", False):
                stream = original_create(*args, **kwargs)
                return _AnthropicStreamWrapper(
                    stream, tracer, trace_id, model, messages, start_time
                )

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
                content = (
                    response.content[0].text
                    if response.content[0].type == "text"
                    else str(response.content[0])
                )

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
