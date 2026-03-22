# agentlearn

A recursive learning framework that gives any AI agent a self-improvement loop with memory. No fine-tuning, no GPUs — just API calls. The output is readable text files (skills, checklists, rules) that humans can inspect, edit, or reject.

## Quickstart

```python
from agentlearn import Engine

engine = Engine()

@engine.trace
def my_agent(task_input, knowledge="", **kwargs):
    from openai import OpenAI
    client = OpenAI()
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
print(f"Promoted {report.candidates_promoted} knowledge items")
```

## How it works

```
Agent runs → Trace (observe) → Analyze (extract lessons) → Validate (test fixes)
    ↑            → Store (knowledge) → Inject (next run) → Agent runs better ─┘
```

1. **Trace** — The `@engine.trace` decorator records every agent run
2. **Analyze** — An LLM reads failed traces and suggests fixes (skills, checklists, rules)
3. **Validate** — Fixes are tested before promotion (human approval in v0.1)
4. **Inject** — On the next run, relevant knowledge is injected into the agent's context

## Before/After: Invoice Processing Agent

We ran an invoice processing agent on 50 invoices with varying edge cases (missing fields, unusual currencies, date formats). After 3 learning cycles, the system promoted 4 knowledge items:

| Metric | Before (no knowledge) | After (4 promoted skills) |
|--------|----------------------|--------------------------|
| Accuracy | 68% (34/50) | 82% (41/50) |
| Common failures | Missing timezone, wrong currency format, null vendor | Addressed by promoted skills |
| Knowledge items | 0 | 4 active |
| Learning cost | — | $1.23 total |

Promoted knowledge examples:
- **[SKILL]** "Before processing an invoice, verify: amount is numeric and positive, currency code is valid ISO 4217, date is not more than 90 days in the past, vendor name is not empty"
- **[ANTI_PATTERN]** "Never assume USD when currency field is missing — check the vendor's country and use their local currency"

These are readable text files. You can inspect, edit, or reject any of them.

## CLI

```bash
agentlearn status                          # Learning progress overview
agentlearn learn                           # Trigger a learning cycle
agentlearn knowledge list                  # List knowledge items
agentlearn knowledge show <id>             # Show details
agentlearn knowledge approve <id>          # Promote candidate to active
agentlearn knowledge audit                 # Check knowledge health
agentlearn traces list --status failure    # List failed traces
agentlearn eval add --task "..."           # Add eval case
agentlearn eval generate --from-traces 20  # Auto-generate eval cases
```

## Install

```bash
pip install agentlearn
```

Requires Python 3.10+ and an OpenAI API key (`OPENAI_API_KEY` env var).

## Design principles

1. **Agent and eval are separated.** The agent can change. The eval cannot. No self-gaming.
2. **Keep or discard is automatic and measurable.** A fix either improves performance without regressions, or it gets rejected.
3. **Output is readable, not opaque.** Learned knowledge is text files that humans can read, edit, approve, or reject. When you swap models, the knowledge transfers.
