"""KnowledgeStore protocol — persists knowledge, traces, and eval cases."""

from typing import Optional, Protocol, runtime_checkable

from ..models import KnowledgeItem


@runtime_checkable
class KnowledgeStore(Protocol):
    def store(self, item: KnowledgeItem) -> str:
        """Store a knowledge item. Returns item_id."""
        ...

    def query(
        self,
        task_context: str,
        tags: Optional[list[str]] = None,
        status: str = "active",
        limit: int = 5,
    ) -> list[KnowledgeItem]:
        """Retrieve relevant knowledge for a given task context."""
        ...

    def update_effectiveness(self, item_id: str, helped: bool) -> None:
        """Track whether injected knowledge helped this run."""
        ...

    def deprecate(self, item_id: str, reason: str) -> None:
        """Mark knowledge as no longer useful."""
        ...

    def list_all(self, status: Optional[str] = None) -> list[KnowledgeItem]:
        """List all knowledge items, optionally filtered by status."""
        ...

    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a single knowledge item by ID."""
        ...

    def export_all(self) -> list[KnowledgeItem]:
        """Export all knowledge for portability/sharing."""
        ...

    def import_items(self, items: list[KnowledgeItem]) -> int:
        """Import knowledge items. Returns count imported."""
        ...
