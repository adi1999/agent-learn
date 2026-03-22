"""Injector protocol — selects and formats knowledge for injection."""

from typing import Protocol, runtime_checkable

from ..models import InjectionResult
from ..store.base import KnowledgeStore


@runtime_checkable
class Injector(Protocol):
    def inject(
        self,
        task_input: str,
        knowledge_store: KnowledgeStore,
    ) -> InjectionResult:
        """Select and format relevant knowledge for injection."""
        ...
