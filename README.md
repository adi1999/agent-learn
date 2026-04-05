# agentlearn

A recursive learning framework that gives any AI agent a self-improvement loop with memory. No fine-tuning, no GPUs — just API calls. The output is readable text (skills, checklists, rules) that humans can inspect, edit, or reject.

Works with any agent framework: LangGraph, CrewAI, Agno, custom async pipelines, or raw SDK calls.

## Quickstart

```python
from agentlearn import Engine
from openai import OpenAI

engine = Engine()
client = OpenAI()

@engine.trace
def my_agent(task_input):
    knowledge = engine.get_knowledge(task_input)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n\n{knowledge}"},
            {"role": "user", "content": task_input},
        ],
    )
    return response.choices[0].message.content

# Run your agent — traces are collected automatically
result = my_agent("What is compound interest on $1000 at 5% for 3 years?")

# After collecting traces, run the learning cycle
report = engine.learn()
```

`@engine.trace` observes your agent (input, output, outcome). `engine.get_knowledge()` returns learned rules for the current task — one line to opt in, remove it to opt out.

## How it works

```
Agent runs → Trace (observe) → Analyze (extract lessons) → Validate (test fixes)
    ↑            → Store (knowledge) → Inject (next run) → Agent runs better ─┘
```

1. **Trace** — `@engine.trace` observes every agent run. Pure observation — doesn't modify your agent.
2. **Analyze** — An LLM reads failed traces and suggests fixes (skills, checklists, rules)
3. **Validate** — Fixes are A/B tested before promotion. Must improve by 5%+ with zero regressions.
4. **Inject** — Knowledge is injected automatically via the SmartInjector. Pinned rules are always present, relevant items are hybrid-searched per query.

## Async and multi-param agents

`@engine.trace` automatically handles async functions and complex signatures:

```python
import asyncio, json

@engine.trace(
    input_extractor=lambda biz_name, **kw: biz_name,
    output_extractor=lambda r: json.dumps(r.get("template", {})),
)
async def website_agent(business_name: str, config: dict = None) -> dict:
    knowledge = await engine.get_knowledge_async(business_name)
    # ... agent logic ...
    return {"template": "...", "metadata": {}}

result = asyncio.run(website_agent("Acme Corp", config={"model": "gpt-4o"}))
```

Or use `auto_inject=True` to skip the manual `get_knowledge()` call:

```python
@engine.trace(auto_inject=True)
async def my_agent(task_input, knowledge=""):
    # 'knowledge' is automatically populated with relevant rules
    return call_llm(task_input, system_prompt=knowledge)
```

## Hybrid search and layered injection

Knowledge retrieval uses **hybrid search** — combining FTS5 keyword matching with embedding cosine similarity via Reciprocal Rank Fusion (RRF). This catches both exact keyword matches and semantically similar items.

Knowledge is injected in two tiers:

- **Pinned items** (core rules) — always injected, regardless of the query
- **Normal items** — hybrid-searched for relevance per query

```python
# Pin a critical rule so it's always injected
engine.knowledge.pin(item_id)

# The agent sees:
# === CORE RULES (always apply) ===
# [SKILL] 8a9f4e2b
# position column is TEXT — use '1' not 1
# === END CORE RULES ===
#
# === RELEVANT KNOWLEDGE (context-matched) ===
# [CHECKLIST] 3c1d6f8a (effectiveness: 92%)
# Before date queries, use TO_DATE(date, 'DD Mon YYYY')
# === END RELEVANT KNOWLEDGE ===
```

## Batch evaluation

Measure your agent's performance with a ground-truth dataset — before and after learning.

```python
engine = Engine(store="./knowledge")

# Import a dataset (CSV or list of dicts)
engine.eval.import_csv("eval_set.csv")

# Benchmark your agent
baseline = engine.evaluate(my_agent, name="before-learning")
print(f"Accuracy: {baseline.accuracy:.1%}, Avg Score: {baseline.avg_score:.3f}")

# ... run engine.learn() ...

# Measure improvement
after = engine.evaluate(my_agent, name="after-learning")
comparison = engine.compare_eval_runs(baseline.run_id, after.run_id)
print(f"Accuracy: {comparison.accuracy_delta:+.1%}")
print(f"Regressions: {len(comparison.regressions)}")
```

### Custom scorers

For domain-specific agents, provide a custom scorer instead of relying on string matching:

```python
def sql_scorer(agent_output: str, task_input: str) -> float:
    """Execute the SQL and compare results."""
    expected = run_golden_sql(task_input)
    actual = extract_sql_result(agent_output)
    return 1.0 if actual == expected else 0.0

report = engine.evaluate(my_agent, scorer=sql_scorer)
```

Scorers can also be set per eval case:

```python
engine.eval.add(
    task_input="Who won the most races in 2019?",
    scorer=lambda output, task: 1.0 if "Hamilton" in output and "11" in output else 0.0,
)
```

## What the agent learns

Knowledge items are readable instructions, not opaque weights:

- **[SKILL]** "Before processing an invoice, verify: amount is numeric and positive, currency code is valid ISO 4217, date is not more than 90 days in the past"
- **[ANTI_PATTERN]** "Never assume USD when currency field is missing — check the vendor's country"
- **[CHECKLIST]** "Before calling any external API: check auth token is not expired, verify endpoint URL, set timeout to 30s"

You can inspect, edit, approve, or reject any of them. When you swap models, the knowledge transfers.

## Validated learning — not blind accumulation

Unlike frameworks that let agents save learnings without verification, agentlearn validates every fix before promotion:

- **Statistical A/B testing** — each candidate fix is tested against a held-out eval set
- **5%+ improvement required** with zero regressions and 80% statistical confidence
- **Effectiveness tracking** — every knowledge item tracks how often it's injected and whether it helps
- **Auto-deprecation** — items that drop below 30% effectiveness are automatically retired
- **Conflict detection** — catches contradictory knowledge before it enters the system

Register your agent for automated validation:

```python
engine.set_agent_runner(lambda task, knowledge_ids: my_agent(task))
report = engine.learn()  # Now validates fixes automatically
```

## CLI

```bash
agentlearn status                          # Learning progress overview
agentlearn learn                           # Trigger a learning cycle

# Knowledge management
agentlearn knowledge list                  # List knowledge items (* = pinned)
agentlearn knowledge show <id>             # Show details
agentlearn knowledge approve <id>          # Promote candidate to active
agentlearn knowledge pin <id>              # Pin item (always injected)
agentlearn knowledge unpin <id>            # Unpin item (back to relevance-searched)
agentlearn knowledge audit                 # Check knowledge health

# Traces
agentlearn traces list --status failure    # List failed traces
agentlearn traces search "timeout"         # Full-text search across traces
agentlearn traces blame <id>              # Find which knowledge caused a failure

# Evaluation
agentlearn eval import eval_set.csv        # Import eval cases from CSV
agentlearn eval run --agent myapp:agent    # Run batch evaluation
agentlearn eval history                    # List past eval runs
agentlearn eval compare <id1> <id2>        # Compare two eval runs

# Other
agentlearn snapshot create --tag v1        # Version your knowledge store
agentlearn ab-report                       # A/B control group performance
```

## Plugin system

All components are swappable via Python Protocols:

| Component | Default | Alternatives |
|-----------|---------|-------------|
| **Tracer** | GenericLLMTracer | — |
| **Analyzer** | LLMAnalyzer | BatchAnalyzer, TieredAnalyzer |
| **Validator** | HumanInLoopValidator | StatisticalValidator |
| **OutcomeSignal** | LLMJudge | CompositeSignal, DeterministicSignal |
| **Injector** | SmartInjector | SimpleInjector, EmbeddingInjector, HybridInjector, CanaryInjector |
| **KnowledgeStore** | LocalStore (SQLite + FTS5) | — |

```python
from agentlearn import Engine
from agentlearn.signals.composite import CompositeSignal
from agentlearn.validators.statistical import StatisticalValidator
from agentlearn.injector.simple import SimpleInjector

engine = Engine(
    signal=CompositeSignal(),
    validator=StatisticalValidator(),
    injector=SimpleInjector(),  # Use simple injector instead of SmartInjector
)
```

## Install

```bash
pip install agentlearn
```

Requires Python 3.10+ and an OpenAI API key (`OPENAI_API_KEY` env var).

## Design principles

1. **Pure observability layer.** `@engine.trace` doesn't modify your agent. Add it, remove it — your agent works the same.
2. **Agent and eval are separated.** The agent can change. The eval cannot. No self-gaming.
3. **Keep or discard is measurable.** A fix either improves performance without regressions, or it gets rejected.
4. **Output is readable, not opaque.** Knowledge is text that humans can read, edit, approve, or reject. When you swap models, the knowledge transfers.
5. **Works with any agent.** Sync or async, single-param or multi-param, string output or structured — extractors adapt the trace to your agent's signature.
