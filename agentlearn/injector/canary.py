"""Canary injector — tests new knowledge on a fraction of runs before full rollout."""

from __future__ import annotations

import random
from typing import Optional

from ..injector.embedding import format_knowledge_block
from ..models import InjectionResult, KnowledgeItem
from ..store.base import KnowledgeStore
from ..utils.logging import get_logger

logger = get_logger("canary_injector")

ESTABLISHED_THRESHOLD = 20  # Items injected >= this many times are "established"


class CanaryInjector:
    """Injects established knowledge to all runs, new knowledge only to a canary percentage.

    Established items (>= 20 injections) are always included.
    Newly promoted items (< 20 injections) are only included in canary runs.
    This protects production from untested knowledge while allowing
    new knowledge to prove itself.
    """

    def __init__(
        self,
        canary_percentage: float = 0.1,
        top_k: int = 5,
        established_threshold: int = ESTABLISHED_THRESHOLD,
    ):
        self.canary_percentage = canary_percentage
        self.top_k = top_k
        self.established_threshold = established_threshold

    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Select knowledge with canary logic."""
        try:
            all_items = knowledge_store.query(
                task_context=task_input,
                status="active",
                limit=self.top_k * 2,  # Fetch more to have both pools
            )
        except Exception as e:
            logger.warning(f"Knowledge query failed: {e}")
            return InjectionResult()

        if not all_items:
            return InjectionResult()

        # Split into established vs new
        established = [i for i in all_items if i.times_injected >= self.established_threshold]
        new_items = [i for i in all_items if i.times_injected < self.established_threshold]

        is_canary = random.random() < self.canary_percentage

        if is_canary and new_items:
            # Canary run: include both established + new
            items = (established + new_items)[: self.top_k]
            logger.debug(
                f"Canary run: injecting {len(established)} established + {len(new_items)} new items"
            )
        else:
            # Normal run: only established items
            items = established[: self.top_k]
            if not items and new_items:
                # No established items yet — include new items anyway
                # (cold start: don't block injection entirely)
                items = new_items[: self.top_k]

        formatted = format_knowledge_block(items)
        item_ids = [item.item_id for item in items]

        return InjectionResult(
            system_prompt_additions=formatted,
            items_injected=item_ids,
        )
