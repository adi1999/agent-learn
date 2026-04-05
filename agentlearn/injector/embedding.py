"""Embedding-based knowledge injector — selects and formats relevant knowledge."""

from typing import Optional

from ..models import InjectionResult, KnowledgeItem
from ..store.base import KnowledgeStore
from ..utils.logging import get_logger

logger = get_logger("injector")


def format_knowledge_block(items: list[KnowledgeItem]) -> str:
    """Format knowledge items into the injection block.

    Reusable by both the injector and CLI display.
    """
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


class EmbeddingInjector:
    """Selects relevant knowledge using embedding similarity and formats for injection."""

    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Select and format relevant knowledge for injection."""
        try:
            if hasattr(knowledge_store, "hybrid_query"):
                items = knowledge_store.hybrid_query(
                    task_context=task_input,
                    status="active",
                    limit=self.top_k,
                )
            else:
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

        formatted = format_knowledge_block(items)
        item_ids = [item.item_id for item in items]

        logger.debug(f"Injecting {len(items)} knowledge items")
        return InjectionResult(
            system_prompt_additions=formatted,
            items_injected=item_ids,
        )
