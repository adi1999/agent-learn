# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e .                              # Install in dev mode
pytest tests/ -v                              # Run all 220 tests
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
- `@engine.trace` — decorator that observes agent function calls (pure observation, never modifies the agent function signature). Supports sync and async functions. Accepts `input_extractor`, `output_extractor`, and `auto_inject` params.
- `engine.get_knowledge(task_input)` / `engine.get_knowledge_async(task_input)` — retrieval of relevant knowledge
- `engine.learn(max_fixes)` — runs a full learning cycle (analyze failed traces → generate fixes → validate → promote)
- `engine.set_agent_runner(fn)` — register agent for automated statistical validation during learn()
- `engine.last_knowledge` — property to access the most recent injection result
- `KnowledgeManager` / `EvalManager` — nested managers for CRUD on knowledge items and eval cases
- `KnowledgeManager.pin(id)` / `.unpin(id)` — pin items so they're always injected

### Plugin system — all components are Python Protocols

The Engine accepts swappable implementations for each component:

| Protocol | Default | Alternatives |
|----------|---------|-------------|
| **Tracer** (`tracers/`) | GenericLLMTracer | — |
| **Analyzer** (`analyzers/`) | LLMAnalyzer | BatchAnalyzer, TieredAnalyzer, ConflictDetector |
| **Validator** (`validators/`) | HumanInLoopValidator | StatisticalValidator |
| **OutcomeSignal** (`signals/`) | LLMJudge | CompositeSignal (deterministic + LLM + disagreement), DeterministicSignal |
| **Injector** (`injector/`) | SmartInjector | SimpleInjector, EmbeddingInjector, HybridInjector, CanaryInjector |
| **KnowledgeStore** (`store/`) | LocalStore (SQLite + WAL + FTS5) | — |

### Injector details

- `SmartInjector` (default) — two-tier: pinned items always injected as "CORE RULES", normal items hybrid-searched (FTS5 + embedding + RRF) as "RELEVANT KNOWLEDGE"
- `SimpleInjector` — dumps all active items sorted by effectiveness (no search, good for <50 items)
- `EmbeddingInjector` — top-k by hybrid search (prefers hybrid_query, falls back to cosine-only)
- `HybridInjector` — progressive disclosure: compact index + `recall_knowledge()` tool for details
- `CanaryInjector` — staged rollout: established items always, new items only in canary % of runs

### Data models (`models.py`)

Pydantic v2 models. Key types: `Trace`, `Step`, `Outcome`, `CandidateFix`, `KnowledgeItem`, `ValidationResult`, `EvalCase`, `InjectionResult`. Knowledge lifecycle: `CANDIDATE → VALIDATED → ACTIVE → DEPRECATED/ARCHIVED`. KnowledgeItem has `priority` field (`"pinned"` | `"normal"`). EvalCase supports optional `scorer: Callable[[str, str], float]` for domain-specific grading.

### Validation philosophy — conservative by design

Zero regressions allowed, 5%+ improvement required, 0.8 confidence threshold, minimum 5 eval cases. Falls back to human approval if eval set is too small or no `agent_runner` is registered.

### Store (`store/local_store.py`)

SQLite with WAL mode, FTS5 full-text search on traces AND knowledge, thread-safe with `threading.Lock`. Tables: `knowledge` (with `priority` column), `traces`, `eval_cases`, `eval_runs`, `embeddings`, `traces_fts`, `knowledge_fts`. Key methods:
- `hybrid_query()` — FTS5 keyword + embedding cosine similarity merged via Reciprocal Rank Fusion
- `list_pinned()` — fetch all pinned active items
- `_keyword_search()` / `_embedding_search()` — internal search primitives
- Auto-migrates older databases (adds `priority` column if missing)

### CLI (`cli.py`)

Click-based. Entry point: `agentlearn` (configured in `pyproject.toml`). Store path via `--store` flag or `AGENTLEARN_STORE` env var (default: `./knowledge`). Includes `knowledge pin/unpin` commands.

## Design Authority

`DESIGN.md` (73KB) is the authoritative specification. Implementation must match it — when in doubt, consult DESIGN.md over inferring from code.

## Dependencies

Core: `pydantic>=2.0`, `openai>=1.0`, `anthropic>=0.30`, `numpy>=1.24`, `click>=8.0`. No heavy deps (no LangChain, scipy, etc.) — implement from scratch if small.
