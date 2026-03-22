"""Human-in-the-loop validator — presents candidate fixes for CLI approval."""

from __future__ import annotations

from typing import Callable, Optional

import click

from ..models import CandidateFix, EvalCase, ValidationResult
from ..utils.logging import get_logger

logger = get_logger("validator")

FIX_TYPE_COLORS = {
    "skill": "green",
    "checklist": "cyan",
    "routing_rule": "yellow",
    "anti_pattern": "red",
    "code_template": "blue",
}


class HumanInLoopValidator:
    """Presents candidate fixes to a human for approval.

    Uses approval_callback if provided (platform-agnostic),
    otherwise falls back to CLI prompts via Click.
    """

    def validate(
        self,
        candidate: CandidateFix,
        eval_set: list[EvalCase],
        agent_runner: Callable[[str, list[str]], str],
        approval_callback: Optional[Callable[[CandidateFix], bool]] = None,
    ) -> ValidationResult:
        """Present a candidate fix for human review."""
        # Use callback if provided (web dashboard, notebook, API, etc.)
        if approval_callback is not None:
            approved = approval_callback(candidate)
            return ValidationResult(
                passed=approved,
                confidence=1.0,
            )

        # Default: CLI prompt
        return self._cli_review(candidate)

    def _cli_review(self, candidate: CandidateFix) -> ValidationResult:
        """Interactive CLI review with approve/reject/skip/edit."""
        self._display_candidate(candidate)

        while True:
            choice = click.prompt(
                click.style("\n  [a]pprove / [r]eject / [s]kip / [e]dit", bold=True),
                type=click.Choice(["a", "r", "s", "e"], case_sensitive=False),
                show_choices=False,
            )

            if choice == "a":
                click.echo(click.style("  Approved.", fg="green", bold=True))
                return ValidationResult(passed=True, confidence=1.0)

            elif choice == "r":
                click.echo(click.style("  Rejected.", fg="red", bold=True))
                return ValidationResult(passed=False, confidence=1.0)

            elif choice == "s":
                click.echo(click.style("  Skipped.", fg="yellow"))
                return ValidationResult(passed=False, confidence=0.0)

            elif choice == "e":
                edited = click.edit(candidate.content)
                if edited and edited.strip():
                    candidate.content = edited.strip()
                    click.echo(click.style("  Content updated. Review again:", fg="cyan"))
                    self._display_candidate(candidate)
                else:
                    click.echo("  No changes made.")

    def _display_candidate(self, candidate: CandidateFix) -> None:
        """Display a candidate fix in the terminal."""
        color = FIX_TYPE_COLORS.get(candidate.fix_type.value, "white")

        click.echo()
        click.echo(click.style("=" * 60, dim=True))
        click.echo(
            click.style(f"  Candidate Fix ", bold=True)
            + click.style(f"[{candidate.fix_type.value.upper()}]", fg=color, bold=True)
        )
        click.echo(click.style("-" * 60, dim=True))
        click.echo(f"  Confidence: {candidate.confidence:.0%}")
        click.echo(f"  Applies when: {candidate.applies_when}")
        click.echo(f"  Source traces: {', '.join(candidate.source_trace_ids[:3])}")
        click.echo()
        click.echo(click.style("  Reasoning:", bold=True))
        click.echo(f"    {candidate.reasoning}")
        click.echo()
        click.echo(click.style("  Fix content:", bold=True))
        for line in candidate.content.split("\n"):
            click.echo(f"    {line}")
        click.echo(click.style("=" * 60, dim=True))
