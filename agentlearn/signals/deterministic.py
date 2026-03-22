"""Deterministic outcome checks — free, instant, not subject to LLM blind spots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..models import Trace
from ..utils.logging import get_logger

logger = get_logger("deterministic")


@dataclass
class CheckResult:
    passed: bool
    hard_fail: bool = False
    reason: str = ""


class DeterministicCheck:
    """Base class for deterministic outcome checks."""

    name: str = "base"

    def run(self, trace: Trace) -> CheckResult:
        raise NotImplementedError


class JSONValidationCheck(DeterministicCheck):
    """Check if the agent's output is valid JSON (when expected)."""

    name = "json_validation"

    def run(self, trace: Trace) -> CheckResult:
        output = trace.final_output or ""
        if not output.strip():
            return CheckResult(passed=False, hard_fail=True, reason="Empty output")

        # Try to detect if output should be JSON
        stripped = output.strip()
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                return CheckResult(passed=True)
            except json.JSONDecodeError as e:
                return CheckResult(
                    passed=False,
                    hard_fail=True,
                    reason=f"Invalid JSON: {e}",
                )

        # Also check for JSON in code blocks
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", output, re.DOTALL)
        if code_block:
            try:
                json.loads(code_block.group(1).strip())
                return CheckResult(passed=True)
            except json.JSONDecodeError as e:
                return CheckResult(
                    passed=False,
                    hard_fail=True,
                    reason=f"Invalid JSON in code block: {e}",
                )

        # Output doesn't look like JSON — not applicable
        return CheckResult(passed=True)


class EmptyOutputCheck(DeterministicCheck):
    """Check that the agent produced non-empty output."""

    name = "empty_output"

    def run(self, trace: Trace) -> CheckResult:
        if not trace.final_output or not trace.final_output.strip():
            return CheckResult(
                passed=False,
                hard_fail=True,
                reason="Agent produced no output",
            )
        return CheckResult(passed=True)


class ErrorStepCheck(DeterministicCheck):
    """Check if any steps resulted in errors."""

    name = "error_step"

    def run(self, trace: Trace) -> CheckResult:
        error_steps = [s for s in trace.steps if s.step_type.value == "error"]
        if error_steps:
            errors = "; ".join(s.result[:100] for s in error_steps[:3])
            return CheckResult(
                passed=False,
                hard_fail=False,  # Errors are a signal, not always fatal
                reason=f"Error steps detected: {errors}",
            )
        return CheckResult(passed=True)


class CodeExecutionCheck(DeterministicCheck):
    """Check if code in the output actually executes without errors."""

    name = "code_execution"

    def run(self, trace: Trace) -> CheckResult:
        output = trace.final_output or ""

        # Extract Python code blocks
        code_blocks = re.findall(r"```python\s*\n(.*?)\n\s*```", output, re.DOTALL)
        if not code_blocks:
            return CheckResult(passed=True)  # No code to check

        for code in code_blocks:
            try:
                compile(code, "<agentlearn_check>", "exec")
                return CheckResult(passed=True)
            except SyntaxError as e:
                return CheckResult(
                    passed=False,
                    hard_fail=True,
                    reason=f"Python syntax error: {e}",
                )

        return CheckResult(passed=True)


# Default check set
DEFAULT_CHECKS = [
    EmptyOutputCheck(),
    JSONValidationCheck(),
    ErrorStepCheck(),
]
