# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e .                              # Install in dev mode
pytest tests/ -v                              # Run all 133 tests
pytest tests/test_engine.py::TestTraceDecorator -v  # Run specific test class
pytest tests/test_engine.py::TestTraceDecorator::test_name -v  # Single test
pytest --cov=agentlearn tests/                # With coverage
ruff check agentlearn/                        # Lint
ruff format agentlearn/                       # Format (line-length: 100)
```

Tests use `pytest-asyncio` in STRICT mode. Mock `get_embedding()` with seeded randomness in fixtures (see `tests/conftest.py`).

## Architecture

**agentlearn** is a recursive learning framework for AI agents. The core loop:

```
Agent runs → Trace → Analyze → Validate → Store knowledge → Inject next run
```

### Engine (`engine.py`) — the orchestrator

Owns the full lifecycle. Key API:
- `@engine.trace` — decorator that observes agent function calls (pure observation, never modifies the agent function signature)
- `engine.get_knowledge(task_input)` — opt-in retrieval of relevant knowledge
- `engine.learn(max_fixes)` — runs a full learning cycle (analyze failed traces → generate fixes → validate → promote)
- `KnowledgeManager` / `EvalManager` — nested managers for CRUD on knowledge items and eval cases

### Plugin system — all components are Python Protocols

The Engine accepts swappable implementations for each component:

| Protocol | Default | Alternatives |
|----------|---------|-------------|
| **Tracer** (`tracers/`) | GenericLLMTracer | — |
| **Analyzer** (`analyzers/`) | LLMAnalyzer | BatchAnalyzer, TieredAnalyzer, ConflictDetector |
| **Validator** (`validators/`) | HumanInLoopValidator | StatisticalValidator |
| **OutcomeSignal** (`signals/`) | LLMJudge | CompositeSignal (deterministic + LLM + disagreement), DeterministicSignal |
| **Injector** (`injector/`) | SimpleInjector | EmbeddingInjector, HybridInjector, CanaryInjector |
| **KnowledgeStore** (`store/`) | LocalStore (SQLite + WAL + FTS5) | — |

### Data models (`models.py`)

Pydantic v2 models. Key types: `Trace`, `Step`, `Outcome`, `CandidateFix`, `KnowledgeItem`, `ValidationResult`, `EvalCase`, `InjectionResult`. Knowledge lifecycle: `CANDIDATE → VALIDATED → ACTIVE → DEPRECATED/ARCHIVED`.

### Validation philosophy — conservative by design

Zero regressions allowed, 5%+ improvement required, 0.8 confidence threshold, minimum 5 eval cases. Falls back to human approval if eval set is too small.

### Store (`store/local_store.py`)

SQLite with WAL mode, FTS5 full-text search on traces, thread-safe with `threading.Lock`. Tables: `knowledge`, `traces`, `eval_cases`, `embeddings`, `traces_fts`.

### CLI (`cli.py`)

Click-based. Entry point: `agentlearn` (configured in `pyproject.toml`). Store path via `--store` flag or `AGENTLEARN_STORE` env var (default: `./knowledge`).

## Design Authority

`DESIGN.md` (73KB) is the authoritative specification. Implementation must match it — when in doubt, consult DESIGN.md over inferring from code.

## Dependencies

Core: `pydantic>=2.0`, `openai>=1.0`, `anthropic>=0.30`, `numpy>=1.24`, `click>=8.0`. No heavy deps (no LangChain, scipy, etc.) — implement from scratch if small.
