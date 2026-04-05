"""Smart injector — always injects pinned items, hybrid-searches the rest."""

from __future__ import annotations

from ..models import InjectionResult, KnowledgeItem
from ..store.base import KnowledgeStore
from ..utils.logging import get_logger

logger = get_logger("smart_injector")


def format_layered_block(
    pinned: list[KnowledgeItem],
    relevant: list[KnowledgeItem],
) -> str:
    """Format knowledge into two tiers: CORE RULES and RELEVANT KNOWLEDGE."""
    if not pinned and not relevant:
        return ""

    lines: list[str] = []

    if pinned:
        lines.append("=== CORE RULES (always apply) ===")
        lines.append("")
        for item in pinned:
            tag = item.fix_type.value.upper()
            lines.append(f"[{tag}] {item.item_id[:8]}")
            lines.append(item.content)
            lines.append("")
        lines.append("=== END CORE RULES ===")
        lines.append("")

    if relevant:
        lines.append("=== RELEVANT KNOWLEDGE (context-matched) ===")
        lines.append("")
        for item in relevant:
            tag = item.fix_type.value.upper()
            eff = f"{item.effectiveness_rate:.0%}" if item.times_injected > 0 else "new"
            lines.append(f"[{tag}] {item.item_id[:8]} (effectiveness: {eff})")
            lines.append(item.content)
            lines.append("")
        lines.append("=== END RELEVANT KNOWLEDGE ===")

    return "\n".join(lines)


class SmartInjector:
    """Two-tier injector: always injects pinned items, hybrid-searches the rest.

    This is the recommended default injector. It ensures critical rules
    are never missed while keeping context focused on relevant items.
    """

    def __init__(self, max_items: int = 10):
        self.max_items = max_items

    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Select and format knowledge with priority tiers."""
        pinned: list[KnowledgeItem] = []
        relevant: list[KnowledgeItem] = []

        # 1. Fetch all pinned active items (always included)
        if hasattr(knowledge_store, "list_pinned"):
            try:
                pinned = knowledge_store.list_pinned(status="active")
            except Exception as e:
                logger.warning(f"Failed to fetch pinned items: {e}")

        # 2. Hybrid search for normal items
        remaining_budget = max(0, self.max_items - len(pinned))
        if remaining_budget > 0:
            try:
                if hasattr(knowledge_store, "hybrid_query"):
                    candidates = knowledge_store.hybrid_query(
                        task_context=task_input,
                        status="active",
                        limit=remaining_budget,
                    )
                elif hasattr(knowledge_store, "query"):
                    candidates = knowledge_store.query(
                        task_context=task_input,
                        status="active",
                        limit=remaining_budget,
                    )
                else:
                    candidates = knowledge_store.list_all(status="active")[:remaining_budget]
                # Exclude pinned items from relevant section
                pinned_ids = {item.item_id for item in pinned}
                relevant = [c for c in candidates if c.item_id not in pinned_ids][:remaining_budget]
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")

        if not pinned and not relevant:
            return InjectionResult()

        all_items = pinned + relevant
        formatted = format_layered_block(pinned, relevant)
        item_ids = [item.item_id for item in all_items]

        logger.debug(f"Injecting {len(pinned)} pinned + {len(relevant)} relevant items")
        return InjectionResult(
            system_prompt_additions=formatted,
            items_injected=item_ids,
        )
