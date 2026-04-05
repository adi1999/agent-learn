"""Tests for the Engine orchestrator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentlearn.engine import Engine, EvalManager, KnowledgeManager
from agentlearn.models import (
    CandidateFix,
    EvalCase,
    FixType,
    InjectionResult,
    KnowledgeItem,
    KnowledgeStatus,
    LearningReport,
    Outcome,
    OutcomeStatus,
    Trace,
    ValidationResult,
)


def mock_get_embedding(text, *args, **kwargs):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1536).astype(np.float32)


@pytest.fixture
def engine(tmp_path):
    """Engine with mocked external dependencies."""
    with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
        e = Engine(
            store=str(tmp_path / "knowledge"),
            model="gpt-4o-mini",
        )
        yield e
        e._store.close()


class TestEngineInit:
    def test_default_components(self, engine):
        assert engine._tracer is not None
        assert engine._analyzer is not None
        assert engine._validator is not None
        assert engine._signal is not None
        assert engine._injector is not None
        assert engine._store is not None

    def test_custom_store_path(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(store=str(tmp_path / "custom_store"))
            assert e._store.path == str(tmp_path / "custom_store")
            e._store.close()

    def test_injection_toggle(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(store=str(tmp_path / "store"), injection_enabled=False)
            assert e.injection_enabled is False
            e._store.close()


class TestTraceDecorator:
    def test_basic_trace(self, engine):
        """Test that the trace decorator runs the function and stores a trace."""
        # Mock the signal to avoid real LLM calls
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace
        def my_agent(task_input):
            return f"Result for: {task_input}"

        result = my_agent("test task")
        assert result == "Result for: test task"

        # Verify trace was stored
        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].task_input == "test task"
        assert traces[0].outcome.status == OutcomeStatus.SUCCESS

    def test_get_knowledge(self, engine):
        """Test that get_knowledge returns relevant knowledge."""
        # Store active knowledge
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Always check for null",
                applies_when="Processing data",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        knowledge = engine.get_knowledge("Process some data")
        # Should return a string (may be empty if embeddings don't match)
        assert isinstance(knowledge, str)

    def test_trace_handles_exception(self, engine):
        """Test that exceptions are recorded and re-raised."""
        engine._signal = MagicMock()

        @engine.trace
        def failing_agent(task_input):
            raise ValueError("Agent crashed")

        with pytest.raises(ValueError, match="Agent crashed"):
            failing_agent("test")

        # Trace should still be stored with failure outcome
        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].outcome.status == OutcomeStatus.FAILURE

    def test_trace_without_injection(self, engine):
        """Test with injection disabled."""
        engine.injection_enabled = False
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace
        def my_agent(task_input):
            # get_knowledge returns "" when injection disabled
            assert engine.get_knowledge(task_input) == ""
            return "done"

        my_agent("test")

    def test_effectiveness_updated(self, engine):
        """Test that effectiveness is updated after traced runs."""
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        # Create active knowledge
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Test knowledge",
                applies_when="Any task",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        # Mock injector to always inject our item
        engine._injector = MagicMock()
        engine._injector.inject.return_value = InjectionResult(
            system_prompt_additions="test knowledge",
            items_injected=[item.item_id],
        )

        @engine.trace
        def my_agent(task_input):
            # Call get_knowledge to trigger injection tracking
            engine.get_knowledge(task_input)
            return "done"

        my_agent("test")

        updated = engine._store.get(item.item_id)
        assert updated.times_injected == 1
        assert updated.times_helped == 1


class TestRunMethod:
    def test_run(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        def my_agent(task_input):
            return f"Result: {task_input}"

        result = engine.run(my_agent, "test")
        assert result == "Result: test"


class TestLearnCycle:
    def test_learn_no_traces(self, engine):
        report = engine.learn()
        assert report.traces_analyzed == 0
        assert report.candidates_generated == 0

    def _seed_eval_cases(self, engine, count=5):
        """Add eval cases to avoid quick-win mode."""
        from agentlearn.models import EvalCase

        for i in range(count):
            engine._store.store_eval_case(EvalCase(task_input=f"Eval task {i}"))

    def test_learn_with_mock_analyzer(self, engine):
        """Test the full learn cycle with mocked components."""
        # Seed eval cases to bypass quick-win mode
        self._seed_eval_cases(engine)

        # Store a failed trace
        trace = Trace(
            task_input="Calculate compound interest",
            final_output="$1050",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.2),
        )
        engine._store.store_trace(trace)

        # Mock analyzer to return a candidate
        candidate = CandidateFix(
            fix_type=FixType.SKILL,
            content="Use compound interest formula",
            applies_when="Interest calculation",
            confidence=0.8,
            reasoning="Used simple interest",
            source_trace_ids=[trace.trace_id],
        )
        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [candidate]

        # Mock validator to auto-approve
        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(
            passed=True, improvement=0.3, confidence=1.0
        )
        engine.set_agent_runner(lambda task, ids: "dummy")

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.traces_analyzed == 1
        assert report.candidates_generated == 1
        assert report.candidates_promoted == 1

        # Knowledge should be stored
        active = engine._store.list_all(status="active")
        assert len(active) == 1
        assert "compound interest" in active[0].content

    def test_learn_rejects_failed_validation(self, engine):
        # Seed eval cases to bypass quick-win mode
        self._seed_eval_cases(engine)

        trace = Trace(
            task_input="test task",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="Bad fix",
                applies_when="test",
                source_trace_ids=[trace.trace_id],
            )
        ]

        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(passed=False)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.candidates_rejected == 1
        assert report.candidates_promoted == 0

    def test_learn_quick_win_mode(self, engine):
        """Test quick-win mode (empty store, no eval cases)."""
        trace = Trace(
            task_input="test",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="Quick win fix",
                applies_when="test",
                confidence=0.8,
                source_trace_ids=[trace.trace_id],
            )
        ]

        # Mock click.confirm to auto-approve
        with (
            patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding),
            patch("agentlearn.engine.click.confirm", return_value=True),
            patch("agentlearn.engine.click.echo"),
        ):
            report = engine.learn()

        assert report.candidates_promoted == 1


class TestKnowledgeManager:
    def test_list_and_show(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Test",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        items = engine.knowledge.list(status="active")
        assert len(items) == 1

        shown = engine.knowledge.show(item.item_id)
        assert shown.content == "Test"

    def test_approve(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Candidate",
                applies_when="test",
                status=KnowledgeStatus.CANDIDATE,
            )
            engine._store.store(item)

        result = engine.knowledge.approve(item.item_id)
        assert result.status == KnowledgeStatus.ACTIVE

    def test_reject(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="To reject",
                applies_when="test",
                status=KnowledgeStatus.CANDIDATE,
            )
            engine._store.store(item)

        result = engine.knowledge.reject(item.item_id)
        assert result.status == KnowledgeStatus.ARCHIVED

    def test_audit(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            # Active item never injected
            engine._store.store(
                KnowledgeItem(
                    fix_type=FixType.SKILL,
                    content="never used",
                    applies_when="test",
                    status=KnowledgeStatus.ACTIVE,
                )
            )

        report = engine.knowledge.audit()
        assert report.total_active == 1
        assert len(report.never_injected) == 1

    def test_import_sandbox(self, engine):
        items = [
            KnowledgeItem(
                fix_type=FixType.SKILL,
                content=f"Imported {i}",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            for i in range(2)
        ]

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            count = engine.knowledge.import_items(items, sandbox=True)

        assert count == 2
        # Should be candidates, not active (sandboxed)
        candidates = engine._store.list_all(status="candidate")
        assert len(candidates) == 2


class TestEvalManager:
    def test_add_and_list(self, engine):
        case = engine.eval.add(
            task_input="What is 2+2?",
            expected_output="4",
            tags=["math"],
        )
        assert case.source == "manual"

        cases = engine.eval.list()
        assert len(cases) == 1

    def test_count(self, engine):
        for i in range(3):
            engine.eval.add(task_input=f"Task {i}")
        assert engine.eval.count() == 3


class TestStatus:
    def test_empty_status(self, engine):
        status = engine.status()
        assert status.knowledge_total == 0
        assert status.traces_total == 0
        assert status.eval_cases == 0
        assert len(status.next_milestones) > 0

    def test_status_with_data(self, engine):
        engine._store.store_trace(
            Trace(
                task_input="s",
                outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.9),
            )
        )
        engine._store.store_trace(
            Trace(
                task_input="f",
                outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
            )
        )

        status = engine.status()
        assert status.traces_total == 2
        assert status.traces_success == 1
        assert status.traces_failure == 1


class TestExtractors:
    """Tests for custom input/output extractors on @engine.trace."""

    def test_custom_input_extractor(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace(input_extractor=lambda name, **kw: f"Business: {name}")
        def my_agent(business_name, template_id=None, config=None):
            return {"template": "result"}

        result = my_agent("Acme Corp", template_id="t1", config={})
        assert result == {"template": "result"}

        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].task_input == "Business: Acme Corp"

    def test_custom_output_extractor(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace(output_extractor=lambda r: r["template"])
        def my_agent(task_input):
            return {"template": "Hello World", "metadata": {"key": "val"}}

        my_agent("test")
        traces = engine._store.get_traces()
        assert traces[0].final_output == "Hello World"

    def test_both_extractors(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace(
            input_extractor=lambda a, b: f"{a}+{b}",
            output_extractor=lambda r: str(r * 2),
        )
        def my_agent(x, y):
            return x + y

        result = my_agent(3, 4)
        assert result == 7  # Original return unmodified

        traces = engine._store.get_traces()
        assert traces[0].task_input == "3+4"
        assert traces[0].final_output == "14"  # out_ext(7) == str(14)

    def test_default_extractors_backward_compatible(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace
        def my_agent(task_input):
            return "simple string"

        my_agent("test task")
        traces = engine._store.get_traces()
        assert traces[0].task_input == "test task"
        assert traces[0].final_output == "simple string"

    def test_engine_level_default_extractors(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(
                store=str(tmp_path / "knowledge"),
                default_input_extractor=lambda *a, **kw: "global_input",
                default_output_extractor=lambda r: "global_output",
            )
            e._signal = MagicMock()
            e._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

            @e.trace
            def agent(x):
                return x

            agent("anything")
            traces = e._store.get_traces()
            assert traces[0].task_input == "global_input"
            assert traces[0].final_output == "global_output"
            e._store.close()

    def test_decorator_extractors_override_engine_defaults(self, tmp_path):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(
                store=str(tmp_path / "knowledge"),
                default_input_extractor=lambda *a, **kw: "engine_default",
            )
            e._signal = MagicMock()
            e._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

            @e.trace(input_extractor=lambda x: f"override:{x}")
            def agent(x):
                return x

            agent("val")
            traces = e._store.get_traces()
            assert traces[0].task_input == "override:val"
            e._store.close()


class TestAsyncTrace:
    """Tests for async agent function tracing."""

    @pytest.mark.asyncio
    async def test_async_trace_basic(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace
        async def my_async_agent(task_input):
            return f"Result for: {task_input}"

        result = await my_async_agent("test task")
        assert result == "Result for: test task"

        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].task_input == "test task"
        assert traces[0].final_output == "Result for: test task"
        assert traces[0].outcome.status == OutcomeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_async_trace_with_extractors(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace(
            input_extractor=lambda name, **kw: name,
            output_extractor=lambda r: r["result"],
        )
        async def my_agent(business_name, config=None):
            return {"result": "generated page", "metadata": {}}

        result = await my_agent("Acme Corp", config={"model": "gpt-4o"})
        assert result == {"result": "generated page", "metadata": {}}

        traces = engine._store.get_traces()
        assert traces[0].task_input == "Acme Corp"
        assert traces[0].final_output == "generated page"

    @pytest.mark.asyncio
    async def test_async_trace_exception(self, engine):
        engine._signal = MagicMock()

        @engine.trace
        async def failing_agent(task_input):
            raise ValueError("Async crash")

        with pytest.raises(ValueError, match="Async crash"):
            await failing_agent("test")

        traces = engine._store.get_traces()
        assert len(traces) == 1
        assert traces[0].outcome.status == OutcomeStatus.FAILURE
        assert "Async crash" in traces[0].outcome.reasoning

    @pytest.mark.asyncio
    async def test_async_get_knowledge(self, engine):
        knowledge = await engine.get_knowledge_async("test")
        assert isinstance(knowledge, str)

    @pytest.mark.asyncio
    async def test_async_get_knowledge_with_injection(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="Test knowledge",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        knowledge = await engine.get_knowledge_async("test")
        assert "Test knowledge" in knowledge

    @pytest.mark.asyncio
    async def test_run_async(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        async def my_agent(task):
            return f"done: {task}"

        result = await engine.run_async(my_agent, "hello")
        assert result == "done: hello"

        traces = engine._store.get_traces()
        assert len(traces) == 1


class TestAgentRunner:
    """Tests for agent_runner registration and validation behavior."""

    def test_set_agent_runner(self, engine):
        runner = MagicMock(return_value="output")
        engine.set_agent_runner(runner)
        assert engine._agent_runner is runner

    def test_constructor_agent_runner(self, tmp_path):
        runner = MagicMock(return_value="output")
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            e = Engine(store=str(tmp_path / "knowledge"), agent_runner=runner)
            assert e._agent_runner is runner
            e._store.close()

    def _seed_eval_cases(self, engine):
        for i in range(6):
            engine._store.store_eval_case(
                __import__("agentlearn.models", fromlist=["EvalCase"]).EvalCase(
                    task_input=f"eval task {i}"
                )
            )

    def test_learn_without_runner_uses_callback(self, engine):
        self._seed_eval_cases(engine)
        trace = Trace(
            task_input="test",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="fix",
                applies_when="test",
                source_trace_ids=[trace.trace_id],
            )
        ]
        engine._approval_callback = MagicMock(return_value=True)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.candidates_promoted == 1
        engine._approval_callback.assert_called_once()

    def test_learn_without_runner_or_callback_rejects(self, engine):
        self._seed_eval_cases(engine)
        trace = Trace(
            task_input="test",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="fix",
                applies_when="test",
                source_trace_ids=[trace.trace_id],
            )
        ]

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        assert report.candidates_rejected == 1
        assert report.candidates_promoted == 0

    def test_learn_with_runner_calls_validator(self, engine):
        self._seed_eval_cases(engine)
        trace = Trace(
            task_input="test",
            outcome=Outcome(status=OutcomeStatus.FAILURE, score=0.1),
        )
        engine._store.store_trace(trace)

        engine._analyzer = MagicMock()
        engine._analyzer.analyze_single.return_value = [
            CandidateFix(
                fix_type=FixType.SKILL,
                content="fix",
                applies_when="test",
                source_trace_ids=[trace.trace_id],
            )
        ]
        engine._validator = MagicMock()
        engine._validator.validate.return_value = ValidationResult(
            passed=True, improvement=0.2, confidence=0.9
        )
        engine.set_agent_runner(lambda task, ids: "output")

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            report = engine.learn()

        engine._validator.validate.assert_called_once()
        assert report.candidates_promoted == 1


class TestStreamingTracer:
    """Tests for streaming support in wrap_openai/wrap_anthropic."""

    def test_openai_stream_wrapper_buffers_content(self):
        from agentlearn.tracers.generic_llm import GenericLLMTracer, _OpenAIStreamWrapper

        tracer = GenericLLMTracer()
        trace_id = tracer.start_trace("test task")

        # Simulate OpenAI streaming chunks
        class FakeChoice:
            def __init__(self, content):
                self.delta = MagicMock(content=content)

        class FakeChunk:
            def __init__(self, content=None, usage=None):
                self.choices = [FakeChoice(content)] if content else []
                self.usage = usage

        chunks = [
            FakeChunk(content="Hello"),
            FakeChunk(content=" "),
            FakeChunk(content="World"),
            FakeChunk(content=None),  # Final chunk
        ]

        wrapper = _OpenAIStreamWrapper(
            iter(chunks), tracer, trace_id, "gpt-4o", [{"role": "user", "content": "hi"}], 0.0
        )

        collected = list(wrapper)
        assert len(collected) == 4

        # Step should be recorded after stream exhaustion
        trace = tracer._active_traces.get(trace_id)
        assert trace is not None
        assert len(trace.steps) == 1
        assert trace.steps[0].result == "Hello World"
        assert trace.steps[0].metadata["streamed"] is True

    def test_openai_stream_wrapper_context_manager(self):
        from agentlearn.tracers.generic_llm import GenericLLMTracer, _OpenAIStreamWrapper

        tracer = GenericLLMTracer()
        trace_id = tracer.start_trace("test")

        class FakeChoice:
            def __init__(self, content):
                self.delta = MagicMock(content=content)

        class FakeChunk:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]
                self.usage = None

        chunks = [FakeChunk("data")]

        wrapper = _OpenAIStreamWrapper(iter(chunks), tracer, trace_id, "gpt-4o", [], 0.0)

        with wrapper as w:
            for _ in w:
                pass

        trace = tracer._active_traces.get(trace_id)
        assert len(trace.steps) == 1
        assert trace.steps[0].result == "data"

    def test_anthropic_stream_wrapper_buffers_content(self):
        from agentlearn.tracers.generic_llm import GenericLLMTracer, _AnthropicStreamWrapper

        tracer = GenericLLMTracer()
        trace_id = tracer.start_trace("test")

        class FakeEvent:
            def __init__(self, event_type, text=None, input_tokens=None, output_tokens=None):
                self.type = event_type
                if event_type == "content_block_delta" and text:
                    self.delta = MagicMock(text=text)
                elif event_type == "message_start" and input_tokens:
                    self.message = MagicMock()
                    self.message.usage = MagicMock(input_tokens=input_tokens)
                elif event_type == "message_delta" and output_tokens:
                    self.usage = MagicMock(output_tokens=output_tokens)

        events = [
            FakeEvent("message_start", input_tokens=10),
            FakeEvent("content_block_delta", text="Hello "),
            FakeEvent("content_block_delta", text="World"),
            FakeEvent("message_delta", output_tokens=5),
        ]

        wrapper = _AnthropicStreamWrapper(
            iter(events), tracer, trace_id, "claude-sonnet-4-20250514", [], 0.0
        )

        collected = list(wrapper)
        assert len(collected) == 4

        trace = tracer._active_traces.get(trace_id)
        assert len(trace.steps) == 1
        assert trace.steps[0].result == "Hello World"
        assert trace.steps[0].metadata["streamed"] is True
        assert trace.steps[0].metadata["input_tokens"] == 10
        assert trace.steps[0].metadata["output_tokens"] == 5


class TestSmartInjector:
    """Tests for SmartInjector with layered context."""

    def test_pinned_items_always_injected(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            # Store a pinned item
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="ALWAYS DO THIS",
                applies_when="everything",
                status=KnowledgeStatus.ACTIVE,
                priority="pinned",
            )
            engine._store.store(item)

        knowledge = engine.get_knowledge("completely unrelated query")
        assert "ALWAYS DO THIS" in knowledge
        assert "CORE RULES" in knowledge

    def test_normal_items_in_relevant_section(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            engine._store.store(
                KnowledgeItem(
                    fix_type=FixType.SKILL,
                    content="specific fix",
                    applies_when="edge case",
                    status=KnowledgeStatus.ACTIVE,
                )
            )

        knowledge = engine.get_knowledge("edge case query")
        if knowledge:
            assert "RELEVANT KNOWLEDGE" in knowledge

    def test_pinned_not_duplicated(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="pinned content",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
                priority="pinned",
            )
            engine._store.store(item)

        knowledge = engine.get_knowledge("test")
        # Should appear exactly once
        assert knowledge.count("pinned content") == 1

    def test_default_injector_is_smart(self, engine):
        from agentlearn.injector.smart import SmartInjector

        assert isinstance(engine._injector, SmartInjector)


class TestPinUnpin:
    """Tests for knowledge pin/unpin management."""

    def test_pin(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="rule",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
            )
            engine._store.store(item)

        result = engine.knowledge.pin(item.item_id)
        assert result is not None
        assert result.priority == "pinned"

        retrieved = engine._store.get(item.item_id)
        assert retrieved.priority == "pinned"

    def test_unpin(self, engine):
        with patch("agentlearn.store.local_store.get_embedding", side_effect=mock_get_embedding):
            item = KnowledgeItem(
                fix_type=FixType.SKILL,
                content="rule",
                applies_when="test",
                status=KnowledgeStatus.ACTIVE,
                priority="pinned",
            )
            engine._store.store(item)

        result = engine.knowledge.unpin(item.item_id)
        assert result is not None
        assert result.priority == "normal"

    def test_pin_nonexistent(self, engine):
        assert engine.knowledge.pin("nonexistent") is None


class TestAutoInject:
    """Tests for auto_inject parameter on @trace."""

    def test_auto_inject_passes_knowledge_kwarg(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        received_knowledge = []

        @engine.trace(auto_inject=True)
        def my_agent(task_input, knowledge=""):
            received_knowledge.append(knowledge)
            return "done"

        my_agent("test")
        assert len(received_knowledge) == 1
        assert isinstance(received_knowledge[0], str)

    def test_auto_inject_without_knowledge_param(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        @engine.trace(auto_inject=True)
        def my_agent(task_input):
            return "done"

        result = my_agent("test")
        assert result == "done"
        # last_knowledge should still be populated
        assert isinstance(engine.last_knowledge, str)

    def test_auto_inject_false_by_default(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        received = []

        @engine.trace
        def my_agent(task_input, knowledge="DEFAULT"):
            received.append(knowledge)
            return "done"

        my_agent("test")
        assert received[0] == "DEFAULT"  # Not overwritten

    @pytest.mark.asyncio
    async def test_auto_inject_async(self, engine):
        engine._signal = MagicMock()
        engine._signal.evaluate.return_value = Outcome(status=OutcomeStatus.SUCCESS, score=0.9)

        received = []

        @engine.trace(auto_inject=True)
        async def my_agent(task_input, knowledge=""):
            received.append(knowledge)
            return "done"

        await my_agent("test")
        assert len(received) == 1
        assert isinstance(received[0], str)


class TestCustomScorer:
    """Tests for custom scorer on eval."""

    def test_case_level_scorer(self):
        from agentlearn.utils.scoring import score_output

        case = EvalCase(
            task_input="What is 2+2?",
            scorer=lambda output, task: 1.0 if "4" in output else 0.0,
        )
        assert score_output("The answer is 4", case) == 1.0
        assert score_output("The answer is 5", case) == 0.0

    def test_global_scorer_fallback(self):
        from agentlearn.utils.scoring import score_output

        case = EvalCase(task_input="test")
        scorer = lambda output, task: 0.75
        assert score_output("anything", case, scorer=scorer) == 0.75

    def test_case_scorer_overrides_global(self):
        from agentlearn.utils.scoring import score_output

        case = EvalCase(
            task_input="test",
            scorer=lambda output, task: 1.0,
        )
        assert score_output("x", case, scorer=lambda o, t: 0.0) == 1.0

    def test_scorer_fallback_to_expected_output(self):
        from agentlearn.utils.scoring import score_output

        case = EvalCase(task_input="test", expected_output="hello")
        assert score_output("hello", case) == 1.0

    def test_batch_evaluator_with_scorer(self):
        from agentlearn.evaluator import BatchEvaluator

        evaluator = BatchEvaluator(
            pass_threshold=0.5,
            scorer=lambda output, task: 0.9 if output else 0.0,
        )
        cases = [EvalCase(task_input="test")]
        report = evaluator.run(lambda t: "result", cases)
        assert report.passed == 1
        assert report.avg_score == 0.9
