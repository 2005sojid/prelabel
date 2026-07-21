"""Durable storage for projects and their results.

Plain ``sqlite3`` — no ORM. The data is two flat tables with one relationship
between them, and an ORM would add a dependency and a layer of indirection to
express what fits in a handful of statements.

**Threading.** FastAPI runs the synchronous handlers in a thread pool and the
project runner works in a thread of its own, so several threads reach this class
at once. Each thread gets its own connection (SQLite connections are not
shareable) and the database runs in WAL mode, which lets readers continue while
a writer commits — exactly the shape of this workload: one writer making
progress, the UI polling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("prelabel.store")

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    source_dir    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new',
    detail        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    model_json    TEXT NOT NULL DEFAULT '{}',
    settings_json TEXT NOT NULL DEFAULT '{}',
    -- Cached project-wide comparison, so reading it does not mean re-matching
    -- every box in the project. Rewritten whenever either set changes.
    comparison_json TEXT NOT NULL DEFAULT '{}',
    -- State of the most recent retrain: progress, metrics, produced weights.
    -- Kept on the project so it survives a restart like everything else here.
    training_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path        TEXT NOT NULL,
    width           INTEGER NOT NULL DEFAULT 0,
    height          INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    detail          TEXT NOT NULL DEFAULT '',
    task            TEXT NOT NULL DEFAULT '',
    inference_ms    REAL NOT NULL DEFAULT 0,
    review_priority REAL NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    detections_json TEXT NOT NULL DEFAULT '[]',
    -- A second annotation set to compare against: an earlier run, another
    -- model, or ground truth imported back from CVAT. Empty until one is set.
    baseline_json   TEXT NOT NULL DEFAULT '[]',
    baseline_count  INTEGER NOT NULL DEFAULT 0,
    -- How much the two sets disagree, so the review queue can lead with it.
    disputed        INTEGER NOT NULL DEFAULT 0,
    agreement       REAL NOT NULL DEFAULT 1.0,
    UNIQUE(project_id, rel_path)
);

"""

#: Created *after* the migration, not with the tables: an index over a column a
#: previous version never had cannot be built until that column exists.
INDEXES = """
CREATE INDEX IF NOT EXISTS items_by_project  ON items(project_id, id);
CREATE INDEX IF NOT EXISTS items_by_status   ON items(project_id, status);
CREATE INDEX IF NOT EXISTS items_by_priority ON items(project_id, review_priority DESC);
CREATE INDEX IF NOT EXISTS items_by_dispute  ON items(project_id, disputed DESC);
"""

#: Project lifecycle. ``new`` has items but has not been run yet.
PROJECT_STATUSES = ("new", "running", "done", "cancelled", "failed")
ITEM_STATUSES = ("pending", "done", "error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Project:
    id: str
    name: str
    source_dir: str
    status: str = "new"
    detail: str = ""
    created_at: str = ""
    updated_at: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    #: Cached result of the last comparison; empty until one has been run.
    comparison: dict[str, Any] = field(default_factory=dict)
    #: State of the most recent retrain; empty until one has been started.
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Item:
    id: int
    rel_path: str
    width: int = 0
    height: int = 0
    status: str = "pending"
    detail: str = ""
    task: str = ""
    inference_ms: float = 0.0
    review_priority: float = 0.0
    detection_count: int = 0
    detections: list[dict[str, Any]] = field(default_factory=list)
    #: The set this item is compared against, if one has been captured.
    baseline: list[dict[str, Any]] = field(default_factory=list)
    baseline_count: int = 0
    #: Cached comparison, so the review queue can sort on it without recomputing.
    disputed: int = 0
    agreement: float = 1.0

    @property
    def name(self) -> str:
        return Path(self.rel_path).name

    @property
    def has_baseline(self) -> bool:
        return self.baseline_count > 0 or bool(self.baseline)

    def to_dict(self, with_detections: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["name"] = self.name
        if not with_detections:
            data.pop("detections")
            data.pop("baseline")
        return data


class Store:
    """Repository over the project database."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)
            connection.executescript(INDEXES)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Bring an older database up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already
        exists, so new columns have to be added explicitly or a database written
        by an earlier version keeps working right up until the first query that
        needs them. Adding a column is cheap and keeps the user's projects.
        """
        project_columns = {row["name"] for row in connection.execute("PRAGMA table_info(projects)")}
        if "comparison_json" not in project_columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN comparison_json TEXT NOT NULL DEFAULT '{}'")
            log.info("Migrated projects database: added projects.comparison_json")
        if "training_json" not in project_columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN training_json TEXT NOT NULL DEFAULT '{}'")
            log.info("Migrated projects database: added projects.training_json")

        existing = {row["name"] for row in connection.execute("PRAGMA table_info(items)")}
        additions = {
            "baseline_json": "TEXT NOT NULL DEFAULT '[]'",
            "baseline_count": "INTEGER NOT NULL DEFAULT 0",
            "disputed": "INTEGER NOT NULL DEFAULT 0",
            "agreement": "REAL NOT NULL DEFAULT 1.0",
        }
        for column, definition in additions.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE items ADD COLUMN {column} {definition}")
                log.info("Migrated projects database: added items.%s", column)

    # -- connections ---------------------------------------------------------

    @property
    def _connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        self._local.connection = connection
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A transaction. Serialised, because SQLite allows one writer at a time."""
        with self._write_lock:
            connection = self._connection
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    # -- projects ------------------------------------------------------------

    def create_project(self, name: str, source_dir: str, settings: dict | None = None) -> Project:
        project = Project(
            id=uuid.uuid4().hex[:12],
            name=name.strip() or "Untitled",
            source_dir=source_dir,
            created_at=_now(),
            updated_at=_now(),
            settings=settings or {},
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO projects (id, name, source_dir, status, created_at, updated_at,"
                " model_json, settings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (project.id, project.name, project.source_dir, project.status,
                 project.created_at, project.updated_at, "{}", json.dumps(project.settings)),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        row = self._connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        rows = self._connection.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [_row_to_project(row) for row in rows]

    def update_project(self, project_id: str, **fields: Any) -> None:
        """Patch a project. Unknown keys are ignored rather than silently written."""
        columns, values = [], []
        for key, value in fields.items():
            if key in ("name", "status", "detail", "source_dir"):
                columns.append(f"{key} = ?")
                values.append(value)
            elif key == "model":
                columns.append("model_json = ?")
                values.append(json.dumps(value))
            elif key == "settings":
                columns.append("settings_json = ?")
                values.append(json.dumps(value))
            elif key == "comparison":
                columns.append("comparison_json = ?")
                values.append(json.dumps(value))
            elif key == "training":
                columns.append("training_json = ?")
                values.append(json.dumps(value))
            else:
                log.debug("Ignoring unknown project field %r", key)
        if not columns:
            return
        columns.append("updated_at = ?")
        values.extend([_now(), project_id])
        with self._connect() as connection:
            connection.execute(f"UPDATE projects SET {', '.join(columns)} WHERE id = ?", values)

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0

    # -- items ---------------------------------------------------------------

    def add_items(self, project_id: str, rel_paths: Sequence[str]) -> int:
        """Register images for a project. Re-adding an existing path is a no-op."""
        with self._connect() as connection:
            cursor = connection.executemany(
                "INSERT OR IGNORE INTO items (project_id, rel_path) VALUES (?, ?)",
                [(project_id, rel_path) for rel_path in rel_paths],
            )
        return cursor.rowcount

    def get_item(self, project_id: str, item_id: int) -> Item | None:
        row = self._connection.execute(
            "SELECT * FROM items WHERE project_id = ? AND id = ?", (project_id, item_id)
        ).fetchone()
        return _row_to_item(row) if row else None

    def pending_items(self, project_id: str, limit: int) -> list[Item]:
        rows = self._connection.execute(
            "SELECT * FROM items WHERE project_id = ? AND status = 'pending' ORDER BY id LIMIT ?",
            (project_id, limit),
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def list_items(
        self,
        project_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
        order: str = "path",
        only: str | None = None,
        search: str | None = None,
        with_detections: bool = True,
    ) -> list[Item]:
        """Page through a project's items.

        ``order='priority'`` is the active-learning view: least-confident first,
        so review time goes where the model is most likely to be wrong.
        """
        clauses = ["project_id = ?"]
        values: list[Any] = [project_id]

        if only == "with":
            clauses.append("detection_count > 0")
        elif only == "without":
            clauses.append("detection_count = 0 AND status = 'done'")
        elif only == "error":
            clauses.append("status = 'error'")
        elif only == "disputed":
            clauses.append("disputed > 0")
        elif only == "agreed":
            clauses.append("disputed = 0 AND baseline_count > 0")

        if search:
            clauses.append("rel_path LIKE ?")
            values.append(f"%{search}%")

        ordering = {
            "path": "rel_path ASC",
            "priority": "review_priority DESC, rel_path ASC",
            "most": "detection_count DESC, rel_path ASC",
            "least": "detection_count ASC, rel_path ASC",
            "disputed": "disputed DESC, agreement ASC, rel_path ASC",
        }.get(order, "rel_path ASC")

        columns = "*" if with_detections else (
            "id, project_id, rel_path, width, height, status, detail, task, "
            "inference_ms, review_priority, detection_count, baseline_count, "
            "disputed, agreement, '[]' AS detections_json, '[]' AS baseline_json"
        )
        values.extend([limit, offset])
        rows = self._connection.execute(
            f"SELECT {columns} FROM items WHERE {' AND '.join(clauses)} "
            f"ORDER BY {ordering} LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def save_result(
        self,
        project_id: str,
        item_id: int,
        *,
        width: int,
        height: int,
        task: str,
        inference_ms: float,
        review_priority: float,
        detections: list[dict],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET status='done', detail='', width=?, height=?, task=?,"
                " inference_ms=?, review_priority=?, detection_count=?, detections_json=?"
                " WHERE project_id=? AND id=?",
                (width, height, task, inference_ms, review_priority, len(detections),
                 json.dumps(detections), project_id, item_id),
            )

    def save_error(self, project_id: str, item_id: int, detail: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET status='error', detail=? WHERE project_id=? AND id=?",
                (detail[:500], project_id, item_id),
            )

    def reset_items(self, project_id: str) -> None:
        """Mark everything pending again — used when the model changes.

        The baseline survives: it is the whole point of a re-run to compare
        against it. The *verdict* does not — ``disputed`` described results that
        no longer exist, and a half-finished re-run would otherwise show stale
        disagreement counts on images it has not reached yet.
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET status='pending', detail='', task='', inference_ms=0,"
                " review_priority=0, detection_count=0, detections_json='[]',"
                " disputed=0, agreement=0"
                " WHERE project_id=?",
                (project_id,),
            )

    # -- comparison ----------------------------------------------------------

    def capture_baseline(self, project_id: str) -> int:
        """Freeze the current annotations as the set to compare against.

        Copies rather than references: the point is to keep what this run said
        while the next one overwrites it.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE items SET baseline_json = detections_json,"
                " baseline_count = detection_count, disputed = 0, agreement = 1.0"
                " WHERE project_id = ? AND status = 'done'",
                (project_id,),
            )
        return cursor.rowcount

    def set_baseline(self, project_id: str, item_id: int, detections: list[dict]) -> None:
        """Put a specific set on one item — used when importing ground truth."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET baseline_json = ?, baseline_count = ?"
                " WHERE project_id = ? AND id = ?",
                (json.dumps(detections), len(detections), project_id, item_id),
            )

    def clear_baseline(self, project_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET baseline_json = '[]', baseline_count = 0,"
                " disputed = 0, agreement = 1.0 WHERE project_id = ?",
                (project_id,),
            )

    def save_comparison(self, project_id: str, item_id: int, disputed: int, agreement: float) -> None:
        """Cache one item's comparison so the queue can sort on it in SQL."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE items SET disputed = ?, agreement = ? WHERE project_id = ? AND id = ?",
                (int(disputed), float(agreement), project_id, item_id),
            )

    def baseline_size(self, project_id: str) -> int:
        """How many items carry a baseline. Zero means nothing to compare."""
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM items WHERE project_id = ? AND baseline_count > 0",
            (project_id,),
        ).fetchone()
        return row["n"] or 0

    def iter_comparable_items(self, project_id: str, chunk: int = 500) -> Iterator[Item]:
        """Stream items that have something on both sides worth comparing.

        An item with no detections *and* no baseline is not a disagreement, so
        including it would only dilute the totals.
        """
        offset = 0
        while True:
            rows = self._connection.execute(
                "SELECT * FROM items WHERE project_id = ? AND status = 'done'"
                " AND (detection_count > 0 OR baseline_count > 0)"
                " ORDER BY id LIMIT ? OFFSET ?",
                (project_id, chunk, offset),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield _row_to_item(row)
            offset += len(rows)

    # -- aggregates ----------------------------------------------------------

    def stats(self, project_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(status='done') AS done,"
            " SUM(status='error') AS failed,"
            " SUM(status='pending') AS pending,"
            " SUM(detection_count) AS detections,"
            " AVG(NULLIF(inference_ms, 0)) AS avg_ms"
            " FROM items WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "done": row["done"] or 0,
            "failed": row["failed"] or 0,
            "pending": row["pending"] or 0,
            "detections": row["detections"] or 0,
            "average_ms": round(row["avg_ms"], 2) if row["avg_ms"] else 0.0,
        }

    def class_counts(self, project_id: str) -> list[dict[str, Any]]:
        """Detections per class across the project.

        Counted in Python rather than SQL because the detections live in a JSON
        column; a dedicated table would be faster but would double the write cost
        of the hot path for a number only the summary needs.
        """
        rows = self._connection.execute(
            "SELECT detections_json FROM items WHERE project_id = ? AND detection_count > 0",
            (project_id,),
        ).fetchall()

        counts: dict[str, int] = {}
        for row in rows:
            for detection in json.loads(row["detections_json"]):
                name = detection.get("class_name", "?")
                counts[name] = counts.get(name, 0) + 1
        return [
            {"class_name": name, "count": count}
            for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ]

    def iter_done_items(self, project_id: str, chunk: int = 500) -> Iterator[Item]:
        """Stream completed items without holding the whole project in memory."""
        offset = 0
        while True:
            rows = self._connection.execute(
                "SELECT * FROM items WHERE project_id = ? AND status = 'done'"
                " ORDER BY rel_path LIMIT ? OFFSET ?",
                (project_id, chunk, offset),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield _row_to_item(row)
            offset += len(rows)


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        source_dir=row["source_dir"],
        status=row["status"],
        detail=row["detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        model=json.loads(row["model_json"] or "{}"),
        settings=json.loads(row["settings_json"] or "{}"),
        comparison=json.loads(row["comparison_json"] or "{}"),
        training=json.loads(row["training_json"] or "{}"),
    )


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        id=row["id"],
        rel_path=row["rel_path"],
        width=row["width"],
        height=row["height"],
        status=row["status"],
        detail=row["detail"],
        task=row["task"],
        inference_ms=row["inference_ms"],
        review_priority=row["review_priority"],
        detection_count=row["detection_count"],
        detections=json.loads(row["detections_json"] or "[]"),
        baseline=json.loads(row["baseline_json"] or "[]"),
        baseline_count=row["baseline_count"],
        disputed=row["disputed"],
        agreement=row["agreement"],
    )
