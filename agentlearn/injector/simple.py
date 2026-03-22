"""Simple injector — injects all active knowledge items. No embeddings, no API calls."""

from __future__ import annotations

from ..models import InjectionResult, KnowledgeItem
from ..store.base import KnowledgeStore
from ..utils.logging import get_logger

logger = get_logger("simple_injector")


def format_knowledge_block(items: list[KnowledgeItem]) -> str:
    """Format knowledge items into the injection block."""
    if not items:
        return ""

    lines = ["=== LEARNED KNOWLEDGE (auto-injected) ===", ""]

    for item in items:
        tag = item.fix_type.value.upper()
        eff = f"{item.effectiveness_rate:.0%}" if item.times_injected > 0 else "new"
        lines.append(f"[{tag}] {item.item_id[:8]} (effectiveness: {eff})")
        lines.append(item.content)
        lines.append("")

    lines.append("=== END LEARNED KNOWLEDGE ===")
    return "\n".join(lines)


class SimpleInjector:
    """Injects all active knowledge items. No embeddings, no API calls.

    This is the default injector. Works perfectly for stores with <50 items.
    For larger stores that need relevance filtering, use EmbeddingInjector.
    """

    def __init__(self, max_items: int = 50):
        self.max_items = max_items

    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Return all active knowledge items formatted as text."""
        try:
            items = knowledge_store.list_all(status="active")
        except Exception as e:
            logger.warning(f"Failed to list knowledge: {e}")
            return InjectionResult()

        if not items:
            return InjectionResult()

        # Cap at max_items (most effective first)
        items = sorted(items, key=lambda i: i.effectiveness_rate, reverse=True)
        items = items[: self.max_items]

        formatted = format_knowledge_block(items)
        item_ids = [item.item_id for item in items]

        logger.debug(f"Injecting {len(items)} knowledge items")
        return InjectionResult(
            system_prompt_additions=formatted,
            items_injected=item_ids,
        )
