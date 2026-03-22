"""Shared LLM utilities: JSON parsing, client singletons."""

import json
import re
from typing import Optional

from .logging import get_logger

logger = get_logger("llm")

_openai_client = None


def get_openai_client():
    """Lazy singleton for shared OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def parse_json_from_llm(text: str) -> Optional[dict]:
    """Parse JSON from LLM output, handling code fences and common issues.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code blocks
    3. Find first { ... } or [ ... ] block
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` or ``` ... ``` blocks
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find first JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}...")
    return None
