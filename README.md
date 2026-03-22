# agentlearn

A recursive learning framework that gives any AI agent a self-improvement loop with memory. No fine-tuning, no GPUs — just API calls. The output is readable text (skills, checklists, rules) that humans can inspect, edit, or reject.

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
4. **Inject** — `engine.get_knowledge()` returns relevant rules. You decide where to put them.

## What the agent learns

Knowledge items are readable instructions, not opaque weights:

- **[SKILL]** "Before processing an invoice, verify: amount is numeric and positive, currency code is valid ISO 4217, date is not more than 90 days in the past"
- **[ANTI_PATTERN]** "Never assume USD when currency field is missing — check the vendor's country"
- **[CHECKLIST]** "Before calling any external API: check auth token is not expired, verify endpoint URL, set timeout to 30s"

You can inspect, edit, approve, or reject any of them. When you swap models, the knowledge transfers.

## CLI

```bash
agentlearn status                          # Learning progress overview
agentlearn learn                           # Trigger a learning cycle
agentlearn knowledge list                  # List knowledge items
agentlearn knowledge show <id>             # Show details
agentlearn knowledge approve <id>          # Promote candidate to active
agentlearn knowledge audit                 # Check knowledge health
agentlearn traces list --status failure    # List failed traces
agentlearn traces search "timeout"         # Full-text search across traces
agentlearn traces blame <id>              # Find which knowledge caused a failure
agentlearn snapshot create --tag v1        # Version your knowledge store
agentlearn ab-report                       # A/B control group performance
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
