"""Hybrid injector — progressive disclosure with always-on index + on-demand full content."""

from __future__ import annotations

from typing import Optional

from ..models import InjectionResult, KnowledgeItem
from ..store.base import KnowledgeStore
from ..utils.logging import get_logger

logger = get_logger("hybrid_injector")

# Tool schema for recall_knowledge — agents can call this to get full content
RECALL_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "recall_knowledge",
        "description": "Retrieve the full content of a knowledge item by its ID. Use this when the knowledge index shows a relevant item and you need the detailed instructions.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The knowledge item ID (first 8 characters are enough)",
                },
            },
            "required": ["item_id"],
        },
    },
}


def format_knowledge_index(items: list[KnowledgeItem]) -> str:
    """Format a compact index of knowledge items (name + applies_when only).

    ~20 tokens per item instead of ~200 for full content.
    """
    if not items:
        return ""

    lines = ["=== AVAILABLE KNOWLEDGE (use recall_knowledge to load details) ===", ""]

    for item in items:
        tag = item.fix_type.value.upper()
        eff = f"{item.effectiveness_rate:.0%}" if item.times_injected > 0 else "new"
        lines.append(f"  [{tag}] {item.item_id[:8]} ({eff}) — {item.applies_when}")

    lines.append("")
    lines.append("=== END KNOWLEDGE INDEX ===")
    return "\n".join(lines)


class HybridInjector:
    """Two-tier knowledge injection with progressive disclosure.

    Tier 1 (always injected): Compact index showing item IDs and trigger conditions.
    Tier 2 (on demand): Agent calls recall_knowledge(item_id) for full content.

    This scales from 5 items to 500+ without blowing up context. For small
    knowledge stores (<10 items), consider using EmbeddingInjector instead
    which injects full content directly.
    """

    def __init__(
        self,
        top_k: int = 10,
        always_inject_globals: bool = True,
    ):
        self.top_k = top_k
        self.always_inject_globals = always_inject_globals
        self._active_items: dict[str, KnowledgeItem] = {}

    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Select relevant knowledge and format as a compact index."""
        try:
            items = knowledge_store.query(
                task_context=task_input,
                status="active",
                limit=self.top_k,
            )
        except Exception as e:
            logger.warning(f"Knowledge query failed: {e}")
            items = []

        if not items:
            return InjectionResult()

        # Cache items for recall_knowledge lookups
        self._active_items = {item.item_id: item for item in items}

        formatted = format_knowledge_index(items)
        item_ids = [item.item_id for item in items]

        logger.debug(f"Injecting index of {len(items)} knowledge items")
        return InjectionResult(
            system_prompt_additions=formatted,
            items_injected=item_ids,
        )

    def recall(self, item_id: str, knowledge_store: Optional[KnowledgeStore] = None) -> str:
        """Recall full content for a knowledge item (Tier 2).

        Called when the agent uses the recall_knowledge tool.
        """
        # Try cached items first
        for full_id, item in self._active_items.items():
            if full_id.startswith(item_id):
                return self._format_full_item(item)

        # Fall back to store lookup
        if knowledge_store is not None:
            items = knowledge_store.list_all(status="active")
            for item in items:
                if item.item_id.startswith(item_id):
                    return self._format_full_item(item)

        return f"Knowledge item '{item_id}' not found."

    def get_tool_schema(self) -> dict:
        """Return the recall_knowledge tool schema for the agent."""
        return RECALL_KNOWLEDGE_TOOL

    def _format_full_item(self, item: KnowledgeItem) -> str:
        tag = item.fix_type.value.upper()
        eff = f"{item.effectiveness_rate:.0%}" if item.times_injected > 0 else "new"
        return (
            f"[{tag}] {item.item_id[:8]} (effectiveness: {eff})\n"
            f"Applies when: {item.applies_when}\n\n"
            f"{item.content}"
        )
