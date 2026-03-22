"""CLI: agentlearn status, learn, knowledge, traces, eval."""

import json
import sys

import click

from .engine import Engine
from .models import KnowledgeItem


def _get_engine(store: str) -> Engine:
    """Create an engine for CLI use (no LLM components needed for most commands)."""
    return Engine(store=store)


@click.group()
@click.option("--store", default="./knowledge", envvar="AGENTLEARN_STORE", help="Knowledge store path")
@click.pass_context
def cli(ctx, store):
    """agentlearn — recursive learning framework for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = store


# === Status ===


@cli.command()
@click.pass_context
def status(ctx):
    """Show engine status and learning progress."""
    engine = _get_engine(ctx.obj["store"])
    s = engine.status()

    click.echo()
    click.echo(click.style("  Learning Progress", bold=True))
    click.echo(f"    Knowledge store: {s.knowledge_total} items "
               f"({s.knowledge_active} active, {s.knowledge_candidate} candidate, "
               f"{s.knowledge_deprecated} deprecated)")
    click.echo(f"    Traces collected: {s.traces_total} "
               f"({s.traces_success} success, {s.traces_failure} failure, "
               f"{s.traces_partial} partial)")
    click.echo(f"    Eval set: {s.eval_cases} cases")
    click.echo(f"    Unanalyzed traces: {s.traces_unanalyzed}")

    if s.baseline_accuracy is not None:
        click.echo()
        click.echo(click.style("  Improvement", bold=True))
        click.echo(f"    Baseline accuracy (first 10 runs): {s.baseline_accuracy:.0%}")
        click.echo(f"    Current accuracy (last 10 runs):   {s.current_accuracy:.0%}  "
                   f"({s.improvement:+.0%})")

    if s.avg_effectiveness > 0:
        click.echo(f"    Avg knowledge effectiveness: {s.avg_effectiveness:.0%}")

    if s.next_milestones:
        click.echo()
        click.echo(click.style("  Next Milestones", bold=True))
        for m in s.next_milestones:
            click.echo(f"    - {m}")

    click.echo()


# === Learn ===


@cli.command()
@click.option("--max-fixes", default=10, help="Max fixes to validate per cycle")
@click.option("--budget", default=None, type=float, help="Max spend for this cycle (USD)")
@click.pass_context
def learn(ctx, max_fixes, budget):
    """Trigger a learning cycle."""
    engine = _get_engine(ctx.obj["store"])
    if budget:
        engine._cost_tracker.budget = budget

    click.echo("Running learning cycle...")
    report = engine.learn(max_fixes=max_fixes)

    click.echo()
    click.echo(click.style("  Learning Report", bold=True))
    click.echo(f"    Traces analyzed: {report.traces_analyzed}")
    click.echo(f"    Candidates generated: {report.candidates_generated}")
    click.echo(f"    Candidates validated: {report.candidates_validated}")
    click.echo(f"    Promoted: {report.candidates_promoted}")
    click.echo(f"    Rejected: {report.candidates_rejected}")
    click.echo(f"    Knowledge deprecated: {report.knowledge_deprecated}")
    click.echo(f"    Cost: ${report.total_cost_usd:.4f}")
    click.echo(f"    Duration: {report.duration_seconds:.1f}s")
    click.echo()


# === Knowledge ===


@cli.group()
def knowledge():
    """Knowledge management commands."""
    pass


@knowledge.command("list")
@click.option("--status", default=None, help="Filter by status (active, candidate, deprecated)")
@click.pass_context
def knowledge_list(ctx, status):
    """List knowledge items."""
    engine = _get_engine(ctx.obj["store"])
    items = engine.knowledge.list(status=status)

    if not items:
        click.echo("  No knowledge items found.")
        return

    click.echo()
    click.echo(f"  {'ID':<10} {'Type':<15} {'Status':<12} {'Eff.':<8} {'Inj.':<6} Content")
    click.echo(f"  {'─' * 10} {'─' * 15} {'─' * 12} {'─' * 8} {'─' * 6} {'─' * 30}")

    for item in items:
        eff = f"{item.effectiveness_rate:.0%}" if item.times_injected > 0 else "n/a"
        content_preview = item.content.replace("\n", " ")[:40]
        click.echo(
            f"  {item.item_id[:8]:<10} {item.fix_type.value:<15} "
            f"{item.status.value:<12} {eff:<8} {item.times_injected:<6} {content_preview}"
        )
    click.echo()


@knowledge.command("show")
@click.argument("item_id")
@click.pass_context
def knowledge_show(ctx, item_id):
    """Show details of a knowledge item."""
    engine = _get_engine(ctx.obj["store"])

    # Support partial ID matching
    items = engine.knowledge.list()
    match = None
    for item in items:
        if item.item_id.startswith(item_id):
            match = item
            break

    if match is None:
        click.echo(f"  Knowledge item '{item_id}' not found.")
        return

    click.echo()
    click.echo(click.style(f"  Knowledge Item: {match.item_id}", bold=True))
    click.echo(f"  Type: {match.fix_type.value}")
    click.echo(f"  Status: {match.status.value}")
    click.echo(f"  Applies when: {match.applies_when}")
    click.echo(f"  Created: {match.created_at.isoformat()}")
    click.echo(f"  Source traces: {', '.join(match.source_trace_ids[:5])}")
    click.echo(f"  Times injected: {match.times_injected}")
    click.echo(f"  Times helped: {match.times_helped}")
    click.echo(f"  Effectiveness: {match.effectiveness_rate:.0%}" if match.times_injected > 0 else "  Effectiveness: n/a")
    if match.validation_improvement is not None:
        click.echo(f"  Validation improvement: {match.validation_improvement:+.2f}")
    click.echo()
    click.echo(click.style("  Content:", bold=True))
    for line in match.content.split("\n"):
        click.echo(f"    {line}")
    click.echo()


@knowledge.command("approve")
@click.argument("item_id")
@click.pass_context
def knowledge_approve(ctx, item_id):
    """Promote a candidate to active."""
    engine = _get_engine(ctx.obj["store"])
    items = engine.knowledge.list()
    full_id = None
    for item in items:
        if item.item_id.startswith(item_id):
            full_id = item.item_id
            break

    if full_id is None:
        click.echo(f"  Item '{item_id}' not found.")
        return

    result = engine.knowledge.approve(full_id)
    if result:
        click.echo(click.style(f"  Approved: {full_id}", fg="green"))
    else:
        click.echo(f"  Cannot approve item (check status).")


@knowledge.command("reject")
@click.argument("item_id")
@click.pass_context
def knowledge_reject(ctx, item_id):
    """Reject a candidate."""
    engine = _get_engine(ctx.obj["store"])
    items = engine.knowledge.list()
    full_id = None
    for item in items:
        if item.item_id.startswith(item_id):
            full_id = item.item_id
            break

    if full_id is None:
        click.echo(f"  Item '{item_id}' not found.")
        return

    engine.knowledge.reject(full_id)
    click.echo(click.style(f"  Rejected: {full_id}", fg="red"))


@knowledge.command("deprecate")
@click.argument("item_id")
@click.pass_context
def knowledge_deprecate(ctx, item_id):
    """Deprecate an active item."""
    engine = _get_engine(ctx.obj["store"])
    items = engine.knowledge.list()
    full_id = None
    for item in items:
        if item.item_id.startswith(item_id):
            full_id = item.item_id
            break

    if full_id is None:
        click.echo(f"  Item '{item_id}' not found.")
        return

    engine.knowledge.deprecate(full_id, reason="manual")
    click.echo(click.style(f"  Deprecated: {full_id}", fg="yellow"))


@knowledge.command("audit")
@click.pass_context
def knowledge_audit(ctx):
    """Audit knowledge health."""
    engine = _get_engine(ctx.obj["store"])
    report = engine.knowledge.audit()

    click.echo()
    click.echo(click.style("  Knowledge Audit", bold=True))
    click.echo(f"    Active items: {report.total_active}")
    click.echo(f"    Avg effectiveness: {report.avg_effectiveness:.0%}" if report.avg_effectiveness > 0 else "    Avg effectiveness: n/a")

    if report.declining_effectiveness:
        click.echo()
        click.echo(click.style("    Declining effectiveness:", fg="red"))
        for d in report.declining_effectiveness:
            click.echo(f"      {d['item_id'][:8]}: {d['effectiveness_rate']:.0%} "
                       f"({d['times_injected']} injections)")

    if report.never_injected:
        click.echo()
        click.echo(click.style("    Never injected:", fg="yellow"))
        for item_id in report.never_injected:
            click.echo(f"      {item_id[:8]}")

    if report.insufficient_data:
        click.echo()
        click.echo(f"    Insufficient data: {len(report.insufficient_data)} items (< 10 injections)")

    click.echo()


@knowledge.command("export")
@click.pass_context
def knowledge_export(ctx):
    """Export all knowledge to JSON (stdout)."""
    engine = _get_engine(ctx.obj["store"])
    items = engine.knowledge.export()
    data = [item.model_dump(mode="json") for item in items]
    click.echo(json.dumps(data, indent=2))


@knowledge.command("import")
@click.pass_context
def knowledge_import(ctx):
    """Import knowledge from JSON (stdin)."""
    engine = _get_engine(ctx.obj["store"])
    data = json.load(sys.stdin)
    items = [KnowledgeItem.model_validate(d) for d in data]
    count = engine.knowledge.import_items(items, sandbox=True)
    click.echo(f"  Imported {count} items as candidates.")


# === Traces ===


@cli.group()
def traces():
    """Trace inspection commands."""
    pass


@traces.command("list")
@click.option("--status", default=None, help="Filter by outcome status (success, failure, partial)")
@click.option("--limit", default=20, help="Max traces to show")
@click.pass_context
def traces_list(ctx, status, limit):
    """List traces."""
    engine = _get_engine(ctx.obj["store"])
    trace_list = engine._store.get_traces(status=status, limit=limit)

    if not trace_list:
        click.echo("  No traces found.")
        return

    click.echo()
    click.echo(f"  {'ID':<10} {'Status':<10} {'Score':<8} {'Steps':<7} {'Cost':<8} Task")
    click.echo(f"  {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 7} {'─' * 8} {'─' * 30}")

    for t in trace_list:
        status_val = t.outcome.status.value if t.outcome else "unknown"
        score = f"{t.outcome.score:.2f}" if t.outcome and t.outcome.score is not None else "n/a"
        task_preview = t.task_input.replace("\n", " ")[:40]
        click.echo(
            f"  {t.trace_id[:8]:<10} {status_val:<10} {score:<8} "
            f"{len(t.steps):<7} ${t.cost_usd:<7.4f} {task_preview}"
        )
    click.echo()


@traces.command("show")
@click.argument("trace_id")
@click.pass_context
def traces_show(ctx, trace_id):
    """Show details of a trace."""
    engine = _get_engine(ctx.obj["store"])

    # Support partial ID
    all_traces = engine._store.get_traces(limit=200)
    match = None
    for t in all_traces:
        if t.trace_id.startswith(trace_id):
            match = t
            break

    if match is None:
        click.echo(f"  Trace '{trace_id}' not found.")
        return

    click.echo()
    click.echo(click.style(f"  Trace: {match.trace_id}", bold=True))
    click.echo(f"  Agent: {match.agent_id}")
    click.echo(f"  Created: {match.created_at.isoformat()}")
    click.echo(f"  Status: {match.outcome.status.value if match.outcome else 'unknown'}")
    click.echo(f"  Score: {match.outcome.score if match.outcome and match.outcome.score is not None else 'n/a'}")
    click.echo(f"  Cost: ${match.cost_usd:.4f}")
    click.echo(f"  Analyzed: {match.analyzed}")
    click.echo()
    click.echo(click.style("  Task Input:", bold=True))
    click.echo(f"    {match.task_input[:500]}")
    click.echo()

    if match.final_output:
        click.echo(click.style("  Output:", bold=True))
        click.echo(f"    {match.final_output[:500]}")
        click.echo()

    if match.outcome and match.outcome.reasoning:
        click.echo(click.style("  Reasoning:", bold=True))
        click.echo(f"    {match.outcome.reasoning}")
        click.echo()

    if match.steps:
        click.echo(click.style(f"  Steps ({len(match.steps)}):", bold=True))
        for i, step in enumerate(match.steps):
            click.echo(f"    {i + 1}. [{step.step_type.value}] {step.decision}")
            click.echo(f"       Result: {step.result[:100]}")
        click.echo()


# === Eval ===


@cli.group("eval")
def eval_group():
    """Eval set management commands."""
    pass


@eval_group.command("list")
@click.pass_context
def eval_list(ctx):
    """List eval cases."""
    engine = _get_engine(ctx.obj["store"])
    cases = engine.eval.list()

    if not cases:
        click.echo("  No eval cases found.")
        return

    click.echo()
    click.echo(f"  {'ID':<10} {'Source':<15} Task")
    click.echo(f"  {'─' * 10} {'─' * 15} {'─' * 40}")

    for case in cases:
        task_preview = case.task_input.replace("\n", " ")[:50]
        click.echo(f"  {case.eval_id[:8]:<10} {case.source:<15} {task_preview}")
    click.echo()


@eval_group.command("add")
@click.option("--task", required=True, help="Task input text")
@click.option("--expected", default=None, help="Expected output (if deterministic)")
@click.pass_context
def eval_add(ctx, task, expected):
    """Add an eval case."""
    engine = _get_engine(ctx.obj["store"])
    case = engine.eval.add(task_input=task, expected_output=expected)
    click.echo(f"  Added eval case: {case.eval_id[:8]}")


@eval_group.command("generate")
@click.option("--from-traces", "from_traces", default=20, type=int, help="Max eval cases to generate from traces")
@click.option("--min-confidence", default=0.85, type=float, help="Min outcome score for promotion")
@click.pass_context
def eval_generate(ctx, from_traces, min_confidence):
    """Auto-generate eval cases from successful traces."""
    engine = _get_engine(ctx.obj["store"])
    count = engine.eval.promote_from_traces(min_confidence=min_confidence, limit=from_traces)
    if count > 0:
        click.echo(f"  Promoted {count} traces to eval cases.")
    else:
        click.echo("  No traces eligible for promotion (need high-confidence successes).")


if __name__ == "__main__":
    cli()
