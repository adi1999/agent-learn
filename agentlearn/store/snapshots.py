"""Knowledge store snapshots — versioning and rollback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..models import KnowledgeItem
from ..utils.logging import get_logger

logger = get_logger("snapshots")


@dataclass
class SnapshotInfo:
    snapshot_id: str
    tag: Optional[str]
    item_count: int
    created_at: str


class SnapshotManager:
    """Version the entire knowledge store state.

    Stores snapshots in a dedicated SQLite table alongside the main store.
    """

    def __init__(self, store):
        self._store = store
        self._init_table()

    def _init_table(self) -> None:
        with self._store._lock:
            self._store._conn.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    tag TEXT,
                    items TEXT NOT NULL,
                    item_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

    def snapshot(self, tag: Optional[str] = None) -> str:
        """Create a point-in-time snapshot of the knowledge store.

        Returns snapshot_id.
        """
        now = datetime.now(timezone.utc)
        snapshot_id = f"snap_{now.strftime('%Y%m%d_%H%M%S')}"

        items = self._store.export_all()
        items_json = json.dumps(
            [item.model_dump(mode="json") for item in items],
            default=str,
        )

        with self._store._lock:
            self._store._conn.execute(
                """INSERT INTO snapshots (snapshot_id, tag, items, item_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (snapshot_id, tag, items_json, len(items), now.isoformat()),
            )
            self._store._conn.commit()

        logger.info(f"Created snapshot {snapshot_id} ({len(items)} items, tag={tag})")
        return snapshot_id

    def restore(self, snapshot_id: str) -> int:
        """Restore knowledge store to a previous snapshot.

        Returns count of items restored.
        """
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT items FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        items_data = json.loads(row["items"])
        items = [KnowledgeItem.model_validate(d) for d in items_data]

        # Clear current knowledge
        with self._store._lock:
            self._store._conn.execute("DELETE FROM knowledge")
            self._store._conn.execute("DELETE FROM embeddings")
            self._store._conn.commit()

        # Import snapshot items
        count = self._store.import_items(items)
        logger.info(f"Restored snapshot {snapshot_id} ({count} items)")
        return count

    def list_snapshots(self) -> list[SnapshotInfo]:
        """List all available snapshots."""
        with self._store._lock:
            rows = self._store._conn.execute(
                "SELECT snapshot_id, tag, item_count, created_at FROM snapshots ORDER BY created_at DESC"
            ).fetchall()

        return [
            SnapshotInfo(
                snapshot_id=row["snapshot_id"],
                tag=row["tag"],
                item_count=row["item_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        with self._store._lock:
            cursor = self._store._conn.execute(
                "DELETE FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            )
            self._store._conn.commit()
        return cursor.rowcount > 0
