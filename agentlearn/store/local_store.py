"""SQLite-backed knowledge store with embedding support."""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from ..models import (
    EvalCase,
    EvalResult,
    EvalRunReport,
    KnowledgeItem,
    KnowledgeStatus,
    Outcome,
    OutcomeStatus,
    Step,
    Trace,
)
from ..utils.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    get_embedding,
    serialize_embedding,
)
from ..utils.logging import get_logger

logger = get_logger("local_store")


class LocalStore:
    """SQLite + embeddings knowledge store.

    Implements the KnowledgeStore protocol and also provides
    trace/eval storage for the engine.
    """

    def __init__(self, path: str = "./knowledge"):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.db_path = os.path.join(path, "agentlearn.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS knowledge (
                    item_id TEXT PRIMARY KEY,
                    fix_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    applies_when TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    source_trace_ids TEXT DEFAULT '[]',
                    analyzer_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    validation_improvement REAL,
                    validation_regressions INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'candidate',
                    times_injected INTEGER DEFAULT 0,
                    times_helped INTEGER DEFAULT 0,
                    effectiveness_rate REAL DEFAULT 0.0,
                    last_injected_at TEXT,
                    last_validated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    agent_id TEXT DEFAULT 'default',
                    parent_trace_id TEXT,
                    task_input TEXT NOT NULL,
                    final_output TEXT,
                    outcome_status TEXT,
                    outcome_score REAL,
                    outcome_reasoning TEXT,
                    steps TEXT DEFAULT '[]',
                    injected_knowledge TEXT DEFAULT '[]',
                    environment TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    cost_usd REAL DEFAULT 0.0,
                    analyzed BOOLEAN DEFAULT FALSE
                );

                CREATE TABLE IF NOT EXISTS eval_cases (
                    eval_id TEXT PRIMARY KEY,
                    task_input TEXT NOT NULL,
                    expected_output TEXT,
                    judge_prompt TEXT,
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT 'auto',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    total_cases INTEGER DEFAULT 0,
                    passed INTEGER DEFAULT 0,
                    failed INTEGER DEFAULT 0,
                    errored INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    avg_score REAL DEFAULT 0.0,
                    min_score REAL DEFAULT 0.0,
                    max_score REAL DEFAULT 0.0,
                    duration_seconds REAL DEFAULT 0.0,
                    pass_threshold REAL DEFAULT 0.7,
                    signal_source TEXT DEFAULT '',
                    tag_stats TEXT DEFAULT '{}',
                    results TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    item_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (item_id) REFERENCES knowledge(item_id)
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge(status);
                CREATE INDEX IF NOT EXISTS idx_traces_analyzed ON traces(analyzed);
                CREATE INDEX IF NOT EXISTS idx_traces_outcome ON traces(outcome_status);

                CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts USING fts5(
                    task_input, final_output, outcome_reasoning,
                    content=traces, content_rowid=rowid
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS traces_ai AFTER INSERT ON traces BEGIN
                    INSERT INTO traces_fts(rowid, task_input, final_output, outcome_reasoning)
                    VALUES (NEW.rowid, NEW.task_input, NEW.final_output, NEW.outcome_reasoning);
                END;

                CREATE TRIGGER IF NOT EXISTS traces_ad AFTER DELETE ON traces BEGIN
                    INSERT INTO traces_fts(traces_fts, rowid, task_input, final_output, outcome_reasoning)
                    VALUES ('delete', OLD.rowid, OLD.task_input, OLD.final_output, OLD.outcome_reasoning);
                END;

                CREATE TRIGGER IF NOT EXISTS traces_au AFTER UPDATE ON traces BEGIN
                    INSERT INTO traces_fts(traces_fts, rowid, task_input, final_output, outcome_reasoning)
                    VALUES ('delete', OLD.rowid, OLD.task_input, OLD.final_output, OLD.outcome_reasoning);
                    INSERT INTO traces_fts(rowid, task_input, final_output, outcome_reasoning)
                    VALUES (NEW.rowid, NEW.task_input, NEW.final_output, NEW.outcome_reasoning);
                END;
            """)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # === KnowledgeStore Protocol ===

    def store(self, item: KnowledgeItem) -> str:
        """Store a knowledge item and its embedding."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO knowledge
                   (item_id, fix_type, content, applies_when, tags, source_trace_ids,
                    analyzer_id, created_at, validation_improvement, validation_regressions,
                    status, times_injected, times_helped, effectiveness_rate,
                    last_injected_at, last_validated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.item_id,
                    item.fix_type.value,
                    item.content,
                    item.applies_when,
                    json.dumps(item.tags),
                    json.dumps(item.source_trace_ids),
                    item.analyzer_id,
                    item.created_at.isoformat(),
                    item.validation_improvement,
                    item.validation_regressions,
                    item.status.value,
                    item.times_injected,
                    item.times_helped,
                    item.effectiveness_rate,
                    item.last_injected_at.isoformat() if item.last_injected_at else None,
                    item.last_validated_at.isoformat() if item.last_validated_at else None,
                ),
            )
            self._conn.commit()

        # Store embedding
        try:
            embed_text = f"{item.applies_when} {item.content}"
            embedding = get_embedding(embed_text)
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (item_id, embedding) VALUES (?, ?)",
                    (item.item_id, serialize_embedding(embedding)),
                )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to store embedding for {item.item_id}: {e}")

        return item.item_id

    def query(
        self,
        task_context: str,
        tags: Optional[list[str]] = None,
        status: str = "active",
        limit: int = 5,
    ) -> list[KnowledgeItem]:
        """Retrieve relevant knowledge using embedding similarity."""
        items = self.list_all(status=status)
        if not items:
            return []

        # Filter by tags if specified
        if tags:
            items = [i for i in items if any(t in i.tags for t in tags)]

        if not items:
            return []

        # Compute embedding similarity
        try:
            query_embedding = get_embedding(task_context)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}. Returning all items.")
            return items[:limit]

        scored = []
        for item in items:
            with self._lock:
                row = self._conn.execute(
                    "SELECT embedding FROM embeddings WHERE item_id = ?",
                    (item.item_id,),
                ).fetchone()

            if row is None:
                continue

            item_embedding = deserialize_embedding(row["embedding"])
            score = cosine_similarity(query_embedding, item_embedding)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def update_effectiveness(self, item_id: str, helped: bool) -> None:
        """Track whether injected knowledge helped this run."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if helped:
                self._conn.execute(
                    """UPDATE knowledge
                       SET times_injected = times_injected + 1,
                           times_helped = times_helped + 1,
                           effectiveness_rate = CAST(times_helped + 1 AS REAL) / (times_injected + 1),
                           last_injected_at = ?
                       WHERE item_id = ?""",
                    (now, item_id),
                )
            else:
                self._conn.execute(
                    """UPDATE knowledge
                       SET times_injected = times_injected + 1,
                           effectiveness_rate = CAST(times_helped AS REAL) / (times_injected + 1),
                           last_injected_at = ?
                       WHERE item_id = ?""",
                    (now, item_id),
                )
            self._conn.commit()

    def deprecate(self, item_id: str, reason: str) -> None:
        """Mark knowledge as deprecated."""
        logger.info(f"Deprecating {item_id}: {reason}")
        with self._lock:
            self._conn.execute(
                "UPDATE knowledge SET status = ? WHERE item_id = ?",
                (KnowledgeStatus.DEPRECATED.value, item_id),
            )
            self._conn.commit()

    def list_all(self, status: Optional[str] = None) -> list[KnowledgeItem]:
        """List all knowledge items, optionally filtered by status."""
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge WHERE status = ?", (status,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM knowledge").fetchall()
        return [self._row_to_knowledge(row) for row in rows]

    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a single knowledge item by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge WHERE item_id = ?", (item_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_knowledge(row)

    def export_all(self) -> list[KnowledgeItem]:
        """Export all knowledge items."""
        return self.list_all()

    def import_items(self, items: list[KnowledgeItem]) -> int:
        """Import knowledge items. Returns count imported."""
        count = 0
        for item in items:
            try:
                self.store(item)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import {item.item_id}: {e}")
        return count

    # === Extended Methods (beyond protocol) ===

    def update_item(self, item: KnowledgeItem) -> None:
        """Update a knowledge item in place."""
        self.store(item)

    def count_by_status(self) -> dict[str, int]:
        """Count knowledge items by status."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) as cnt FROM knowledge GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    # === Trace Storage ===

    def store_trace(self, trace: Trace) -> str:
        """Store a trace."""
        steps_json = json.dumps([s.model_dump(mode="json") for s in trace.steps])
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, agent_id, parent_trace_id, task_input, final_output,
                    outcome_status, outcome_score, outcome_reasoning,
                    steps, injected_knowledge, environment, created_at, cost_usd, analyzed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.trace_id,
                    trace.agent_id,
                    trace.parent_trace_id,
                    trace.task_input,
                    trace.final_output,
                    trace.outcome.status.value if trace.outcome else None,
                    trace.outcome.score if trace.outcome else None,
                    trace.outcome.reasoning if trace.outcome else None,
                    steps_json,
                    json.dumps(trace.injected_knowledge),
                    json.dumps(trace.environment),
                    trace.created_at.isoformat(),
                    trace.cost_usd,
                    trace.analyzed,
                ),
            )
            self._conn.commit()
        return trace.trace_id

    def get_unanalyzed_traces(
        self, limit: int = 50, prioritize_failures: bool = True
    ) -> list[Trace]:
        """Fetch traces that haven't been analyzed yet."""
        order = "outcome_status ASC" if prioritize_failures else "created_at DESC"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM traces WHERE analyzed = 0 ORDER BY {order} LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def mark_trace_analyzed(self, trace_id: str) -> None:
        """Mark a trace as analyzed."""
        with self._lock:
            self._conn.execute("UPDATE traces SET analyzed = 1 WHERE trace_id = ?", (trace_id,))
            self._conn.commit()

    def get_traces(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Trace]:
        """Get traces, optionally filtered by outcome status."""
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM traces WHERE outcome_status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a single trace by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_trace(row)

    def search_traces(self, query: str, limit: int = 50) -> list[Trace]:
        """Full-text search across traces using FTS5."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT traces.* FROM traces_fts
                   JOIN traces ON traces.rowid = traces_fts.rowid
                   WHERE traces_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def count_traces(self) -> dict[str, int]:
        """Count traces by outcome status."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT outcome_status, COUNT(*) as cnt FROM traces GROUP BY outcome_status"
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*) as cnt FROM traces").fetchone()
            unanalyzed = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM traces WHERE analyzed = 0"
            ).fetchone()
        result = {row["outcome_status"] or "unknown": row["cnt"] for row in rows}
        result["total"] = total["cnt"]
        result["unanalyzed"] = unanalyzed["cnt"]
        return result

    # === Eval Case Storage ===

    def store_eval_case(self, case: EvalCase) -> str:
        """Store an eval case."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO eval_cases
                   (eval_id, task_input, expected_output, judge_prompt, tags, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    case.eval_id,
                    case.task_input,
                    case.expected_output,
                    case.judge_prompt,
                    json.dumps(case.tags),
                    case.source,
                    case.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        return case.eval_id

    def list_eval_cases(self, tags: Optional[list[str]] = None) -> list[EvalCase]:
        """List eval cases, optionally filtered by tags."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM eval_cases").fetchall()
        cases = [self._row_to_eval_case(row) for row in rows]
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        return cases

    def count_eval_cases(self) -> int:
        """Count eval cases."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM eval_cases").fetchone()
        return row["cnt"]

    # === Eval Run Storage ===

    def store_eval_run(self, report: EvalRunReport) -> str:
        """Store a batch eval run report."""
        results_json = json.dumps([r.model_dump(mode="json") for r in report.results])
        tag_stats_json = json.dumps(report.tag_stats)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO eval_runs
                   (run_id, name, created_at, total_cases, passed, failed, errored,
                    accuracy, avg_score, min_score, max_score, duration_seconds,
                    pass_threshold, signal_source, tag_stats, results)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.run_id,
                    report.name,
                    report.created_at.isoformat(),
                    report.total_cases,
                    report.passed,
                    report.failed,
                    report.errored,
                    report.accuracy,
                    report.avg_score,
                    report.min_score,
                    report.max_score,
                    report.duration_seconds,
                    report.pass_threshold,
                    report.signal_source,
                    tag_stats_json,
                    results_json,
                ),
            )
            self._conn.commit()
        return report.run_id

    def list_eval_runs(self, limit: int = 20) -> list[EvalRunReport]:
        """List eval runs, most recent first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_eval_run(row) for row in rows]

    def get_eval_run(self, run_id: str) -> Optional[EvalRunReport]:
        """Get a single eval run by ID (supports partial ID match)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                row = self._conn.execute(
                    "SELECT * FROM eval_runs WHERE run_id LIKE ?", (f"{run_id}%",)
                ).fetchone()
        if row is None:
            return None
        return self._row_to_eval_run(row)

    # === Serialization Helpers ===

    def _row_to_knowledge(self, row: sqlite3.Row) -> KnowledgeItem:
        return KnowledgeItem(
            item_id=row["item_id"],
            fix_type=row["fix_type"],
            content=row["content"],
            applies_when=row["applies_when"],
            tags=json.loads(row["tags"]),
            source_trace_ids=json.loads(row["source_trace_ids"]),
            analyzer_id=row["analyzer_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            validation_improvement=row["validation_improvement"],
            validation_regressions=row["validation_regressions"],
            status=row["status"],
            times_injected=row["times_injected"],
            times_helped=row["times_helped"],
            effectiveness_rate=row["effectiveness_rate"],
            last_injected_at=datetime.fromisoformat(row["last_injected_at"])
            if row["last_injected_at"]
            else None,
            last_validated_at=datetime.fromisoformat(row["last_validated_at"])
            if row["last_validated_at"]
            else None,
        )

    def _row_to_trace(self, row: sqlite3.Row) -> Trace:
        outcome = None
        if row["outcome_status"]:
            outcome = Outcome(
                status=OutcomeStatus(row["outcome_status"]),
                score=row["outcome_score"],
                reasoning=row["outcome_reasoning"],
            )

        steps_data = json.loads(row["steps"])
        steps = [Step.model_validate(s) for s in steps_data]

        return Trace(
            trace_id=row["trace_id"],
            agent_id=row["agent_id"],
            parent_trace_id=row["parent_trace_id"],
            task_input=row["task_input"],
            final_output=row["final_output"],
            outcome=outcome,
            steps=steps,
            injected_knowledge=json.loads(row["injected_knowledge"]),
            environment=json.loads(row["environment"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            cost_usd=row["cost_usd"],
            analyzed=bool(row["analyzed"]),
        )

    def _row_to_eval_case(self, row: sqlite3.Row) -> EvalCase:
        return EvalCase(
            eval_id=row["eval_id"],
            task_input=row["task_input"],
            expected_output=row["expected_output"],
            judge_prompt=row["judge_prompt"],
            tags=json.loads(row["tags"]),
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_eval_run(self, row: sqlite3.Row) -> EvalRunReport:
        results_data = json.loads(row["results"])
        results = [EvalResult.model_validate(r) for r in results_data]
        return EvalRunReport(
            run_id=row["run_id"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            total_cases=row["total_cases"],
            passed=row["passed"],
            failed=row["failed"],
            errored=row["errored"],
            accuracy=row["accuracy"],
            avg_score=row["avg_score"],
            min_score=row["min_score"],
            max_score=row["max_score"],
            duration_seconds=row["duration_seconds"],
            pass_threshold=row["pass_threshold"],
            signal_source=row["signal_source"],
            tag_stats=json.loads(row["tag_stats"]),
            results=results,
        )
