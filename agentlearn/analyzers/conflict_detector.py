"""Conflict detector — finds contradictions between knowledge items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import KnowledgeItem
from ..utils.embeddings import cosine_similarity, get_embedding
from ..utils.llm import get_openai_client, parse_json_from_llm
from ..utils.logging import get_logger

logger = get_logger("conflict_detector")

CONTRADICTION_PROMPT = """You are checking if two knowledge items contradict each other.
They have similar trigger conditions (both apply in similar situations).

Item A:
{content_a}

Item B:
{content_b}

Do these two items give contradictory instructions? If someone followed both,
would they get conflicting guidance?

Output JSON:
{{
  "is_contradictory": true/false,
  "reason": "explanation of the contradiction (or why they're compatible)",
  "resolution": "how to scope them to avoid conflict (if contradictory)"
}}"""


@dataclass
class Conflict:
    new_item_id: str
    existing_item_id: str
    trigger_similarity: float
    reason: str
    resolution: str = ""


@dataclass
class ConflictReport:
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


class ConflictDetector:
    """Detects when new knowledge contradicts existing knowledge.

    Two-pass approach:
    1. Embedding similarity on trigger conditions (cheap, fast)
    2. LLM contradiction check only for high-similarity pairs (expensive, accurate)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        use_llm: bool = True,
        model: str = "gpt-4o-mini",
    ):
        self.similarity_threshold = similarity_threshold
        self.use_llm = use_llm
        self.model = model

    def check_conflicts(
        self, new_item: KnowledgeItem, store
    ) -> list[Conflict]:
        """Check if new_item conflicts with any active knowledge."""
        active_items = store.list_all(status="active")
        if not active_items:
            return []

        conflicts = []

        try:
            new_embedding = get_embedding(new_item.applies_when)
        except Exception as e:
            logger.warning(f"Failed to embed new item trigger: {e}")
            return []

        for existing in active_items:
            if existing.item_id == new_item.item_id:
                continue

            # Pass 1: Check trigger similarity
            try:
                existing_embedding = get_embedding(existing.applies_when)
                similarity = cosine_similarity(new_embedding, existing_embedding)
            except Exception:
                continue

            if similarity < self.similarity_threshold:
                continue

            # Pass 2: LLM contradiction check (only for similar triggers)
            if self.use_llm:
                conflict = self._llm_check(new_item, existing, similarity)
                if conflict:
                    conflicts.append(conflict)
            else:
                # Without LLM, flag all high-similarity pairs for manual review
                conflicts.append(Conflict(
                    new_item_id=new_item.item_id,
                    existing_item_id=existing.item_id,
                    trigger_similarity=similarity,
                    reason=f"High trigger similarity ({similarity:.0%}) — manual review needed",
                ))

        return conflicts

    def check_all_conflicts(self, store) -> list[Conflict]:
        """Check all active items against each other for conflicts."""
        active = store.list_all(status="active")
        if len(active) < 2:
            return []

        conflicts = []
        checked = set()

        for i, item_a in enumerate(active):
            for item_b in active[i + 1 :]:
                pair_key = tuple(sorted([item_a.item_id, item_b.item_id]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                try:
                    emb_a = get_embedding(item_a.applies_when)
                    emb_b = get_embedding(item_b.applies_when)
                    similarity = cosine_similarity(emb_a, emb_b)
                except Exception:
                    continue

                if similarity < self.similarity_threshold:
                    continue

                if self.use_llm:
                    conflict = self._llm_check(item_a, item_b, similarity)
                    if conflict:
                        conflicts.append(conflict)
                else:
                    conflicts.append(Conflict(
                        new_item_id=item_a.item_id,
                        existing_item_id=item_b.item_id,
                        trigger_similarity=similarity,
                        reason=f"High trigger similarity ({similarity:.0%})",
                    ))

        return conflicts

    def _llm_check(
        self, item_a: KnowledgeItem, item_b: KnowledgeItem, similarity: float
    ) -> Optional[Conflict]:
        """Use LLM to check if two items actually contradict."""
        prompt = CONTRADICTION_PROMPT.format(
            content_a=item_a.content,
            content_b=item_b.content,
        )

        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            parsed = parse_json_from_llm(response.choices[0].message.content or "")
            if parsed and parsed.get("is_contradictory"):
                return Conflict(
                    new_item_id=item_a.item_id,
                    existing_item_id=item_b.item_id,
                    trigger_similarity=similarity,
                    reason=parsed.get("reason", "LLM detected contradiction"),
                    resolution=parsed.get("resolution", ""),
                )

        except Exception as e:
            logger.warning(f"LLM conflict check failed: {e}")

        return None
