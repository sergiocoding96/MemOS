"""
SQLite-backed durable retry queue with exponential backoff and dead-letter table.

Background (Bug 2, 2026-04-26 storage audit): the prior async write path was
fire-and-forget — when Qdrant or Neo4j was briefly unreachable during the
scheduler's structured-memory extraction, the task was marked "failed" and
the extraction silently disappeared. The caller had already received HTTP 200.

This module replaces that with at-least-once delivery:
    queue.enqueue(payload)         # durable; survives process restart
    queue.start_worker(handler)    # background thread drains pending rows
    handler(payload) -> None       # raise on failure to trigger retry

Retry policy (defaults; configurable on the queue):
    - max_attempts = 10
    - backoff = min(60, 2 ** attempt) seconds, starting at 1s
    - on max_attempts failure → row is moved to the `dead_letter` table

Logging contract:
    - every retry attempt logs at INFO with (queue_id, attempt, last_error)
    - dead-letter writes log at WARNING

The queue is intentionally simple: a single SQLite file, WAL mode, one
worker thread per process. SQLite is already a hard dependency in the v1
deployment, so this adds no new infra.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memos.log import get_logger


logger = get_logger(__name__)

# Default location: ~/.hermes/state/retry_queue.sqlite if HERMES_HOME is set,
# otherwise a tmp path. The deployment can override via MEMOS_RETRY_QUEUE_PATH.
_DEFAULT_HOME = os.path.expanduser("~/.hermes/state")
DEFAULT_QUEUE_PATH = os.environ.get(
    "MEMOS_RETRY_QUEUE_PATH",
    os.path.join(_DEFAULT_HOME, "retry_queue.sqlite"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending (
    id              TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL,
    next_attempt_at REAL NOT NULL,
    last_error      TEXT,
    enqueued_at     REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_due ON pending(next_attempt_at);

CREATE TABLE IF NOT EXISTS dead_letter (
    id              TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    attempts        INTEGER NOT NULL,
    last_error      TEXT,
    enqueued_at     REAL NOT NULL,
    dead_at         REAL NOT NULL
);
"""


class RetryAbort(Exception):
    """Raise from a queue handler to skip retry and dead-letter immediately.

    The queue's default policy is "exception → reschedule with backoff". That
    is correct for transient dependency outages but wrong for two cases that
    only the handler can detect:
      - the underlying resource is gone for good (e.g. mem_cube was deleted
        while the row was in the queue), so retrying just burns attempts
      - the failure is a programming-class error surfaced during retry,
        which would dead-letter eventually but should fail loudly now

    The string passed in is recorded as the dead-letter `last_error`.
    """


def _now() -> float:
    return time.time()


def _backoff_seconds(attempt: int, *, initial: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff: 1, 2, 4, 8, 16, 32, 60, 60, ..."""
    if attempt < 1:
        return 0.0
    return min(cap, initial * (2 ** (attempt - 1)))


@dataclass
class QueueRow:
    id: str
    label: str
    payload: dict[str, Any]
    attempt: int
    max_attempts: int
    last_error: str | None


class RetryQueue:
    """SQLite-backed durable retry queue."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        max_attempts: int = 10,
        poll_interval_s: float = 1.0,
        backoff_initial_s: float = 1.0,
        backoff_cap_s: float = 60.0,
    ) -> None:
        self.db_path = db_path or DEFAULT_QUEUE_PATH
        self.max_attempts = max_attempts
        self.poll_interval_s = poll_interval_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_cap_s = backoff_cap_s

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()  # serialize writes; SQLite + threads
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        self._init_schema()

    # ──────────────────────────────────────────────────────────────────────
    # Schema / connection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _txn(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
            finally:
                conn.close()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def enqueue(
        self,
        label: str,
        payload: dict[str, Any],
        *,
        max_attempts: int | None = None,
        delay_s: float = 0.0,
    ) -> str:
        """Add a row to the pending queue.

        Returns the row id (UUID). The row is durable across process restarts.
        """
        row_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, default=str)
        now = _now()
        with self._lock, self._txn() as conn:
            conn.execute(
                """
                INSERT INTO pending
                (id, label, payload_json, attempt, max_attempts,
                 next_attempt_at, last_error, enqueued_at, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, NULL, ?, ?)
                """,
                (
                    row_id,
                    label,
                    payload_json,
                    max_attempts if max_attempts is not None else self.max_attempts,
                    now + delay_s,
                    now,
                    now,
                ),
            )
        logger.info(f"[RetryQueue] enqueued id={row_id} label={label}")
        return row_id

    def pending_count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        finally:
            conn.close()

    def dead_letter_count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
        finally:
            conn.close()

    def list_dead_letter(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, label, payload_json, attempts, last_error, "
                "enqueued_at, dead_at FROM dead_letter ORDER BY dead_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "label": r["label"],
                    "payload": json.loads(r["payload_json"]),
                    "attempts": r["attempts"],
                    "last_error": r["last_error"],
                    "enqueued_at": datetime.fromtimestamp(
                        r["enqueued_at"], tz=timezone.utc
                    ).isoformat(),
                    "dead_at": datetime.fromtimestamp(
                        r["dead_at"], tz=timezone.utc
                    ).isoformat(),
                }
            )
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Worker loop
    # ──────────────────────────────────────────────────────────────────────

    def start_worker(
        self,
        handler: Callable[[str, dict[str, Any]], None],
        *,
        thread_name: str = "memos-retry-queue",
    ) -> None:
        """Spawn a daemon thread that drains pending rows.

        The handler is called as `handler(label, payload)`. If it raises,
        the row is rescheduled with exponential backoff. After max_attempts,
        the row is moved to dead_letter and a WARNING is logged.
        """
        if self._worker and self._worker.is_alive():
            logger.warning("[RetryQueue] worker already running; ignoring start_worker()")
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(handler,),
            name=thread_name,
            daemon=True,
        )
        self._worker.start()
        logger.info(f"[RetryQueue] worker started ({thread_name}) db={self.db_path}")

    def stop_worker(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=timeout)
            self._worker = None
            logger.info("[RetryQueue] worker stopped")

    def _worker_loop(self, handler: Callable[[str, dict[str, Any]], None]) -> None:
        while not self._stop.is_set():
            try:
                row = self._claim_due_row()
                if row is None:
                    # nothing due — sleep poll_interval
                    self._stop.wait(timeout=self.poll_interval_s)
                    continue
                self._dispatch_one(row, handler)
            except Exception as e:  # defensive — never let the worker die
                logger.error(f"[RetryQueue] worker iteration error: {e}", exc_info=True)
                self._stop.wait(timeout=self.poll_interval_s)

    def drain_once(
        self,
        handler: Callable[[str, dict[str, Any]], None],
        *,
        max_rows: int = 100,
    ) -> int:
        """Process up to `max_rows` due rows in the calling thread.

        Useful for tests and for running the queue from a cron job rather
        than a long-lived worker thread. Returns the number of rows handled.
        """
        processed = 0
        for _ in range(max_rows):
            row = self._claim_due_row()
            if row is None:
                break
            self._dispatch_one(row, handler)
            processed += 1
        return processed

    # ──────────────────────────────────────────────────────────────────────
    # Internal: claim + dispatch + reschedule
    # ──────────────────────────────────────────────────────────────────────

    def _claim_due_row(self) -> QueueRow | None:
        """Return the oldest due row, or None. Does not lock the row in DB
        (single-worker assumption); the row is updated on success/failure."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT id, label, payload_json, attempt, max_attempts, last_error "
                    "FROM pending WHERE next_attempt_at <= ? "
                    "ORDER BY next_attempt_at ASC LIMIT 1",
                    (_now(),),
                )
                r = cur.fetchone()
                if r is None:
                    return None
                return QueueRow(
                    id=r["id"],
                    label=r["label"],
                    payload=json.loads(r["payload_json"]),
                    attempt=r["attempt"],
                    max_attempts=r["max_attempts"],
                    last_error=r["last_error"],
                )
            finally:
                conn.close()

    def _dispatch_one(
        self,
        row: QueueRow,
        handler: Callable[[str, dict[str, Any]], None],
    ) -> None:
        attempt_num = row.attempt + 1
        try:
            handler(row.label, row.payload)
        except RetryAbort as ra:
            err = f"RetryAbort: {ra}"
            self._move_to_dead_letter(row, err)
            logger.warning(
                f"[RetryQueue] dead-letter (abort) id={row.id} label={row.label} "
                f"attempts={attempt_num} last_error={err}"
            )
            return
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.info(
                f"[RetryQueue] retry id={row.id} label={row.label} "
                f"attempt={attempt_num}/{row.max_attempts} last_error={err}"
            )
            if attempt_num >= row.max_attempts:
                self._move_to_dead_letter(row, err)
                logger.warning(
                    f"[RetryQueue] dead-letter id={row.id} label={row.label} "
                    f"attempts={attempt_num} last_error={err}"
                )
            else:
                self._reschedule(row.id, attempt_num, err)
            return

        # success
        self._delete(row.id)
        logger.info(
            f"[RetryQueue] success id={row.id} label={row.label} attempt={attempt_num}"
        )

    def _reschedule(self, row_id: str, attempt: int, error: str) -> None:
        delay = _backoff_seconds(
            attempt, initial=self.backoff_initial_s, cap=self.backoff_cap_s
        )
        with self._lock, self._txn() as conn:
            conn.execute(
                "UPDATE pending SET attempt=?, last_error=?, next_attempt_at=?, updated_at=? "
                "WHERE id=?",
                (attempt, error, _now() + delay, _now(), row_id),
            )

    def _delete(self, row_id: str) -> None:
        with self._lock, self._txn() as conn:
            conn.execute("DELETE FROM pending WHERE id=?", (row_id,))

    def _move_to_dead_letter(self, row: QueueRow, error: str) -> None:
        with self._lock, self._txn() as conn:
            r = conn.execute(
                "SELECT enqueued_at FROM pending WHERE id=?", (row.id,)
            ).fetchone()
            enqueued_at = r["enqueued_at"] if r else _now()
            conn.execute(
                """
                INSERT INTO dead_letter
                (id, label, payload_json, attempts, last_error, enqueued_at, dead_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.id,
                    row.label,
                    json.dumps(row.payload, default=str),
                    row.attempt + 1,
                    error,
                    enqueued_at,
                    _now(),
                ),
            )
            conn.execute("DELETE FROM pending WHERE id=?", (row.id,))

    # ──────────────────────────────────────────────────────────────────────
    # Maintenance
    # ──────────────────────────────────────────────────────────────────────

    def requeue_dead_letter(self, row_id: str) -> bool:
        """Move a dead-letter row back into pending with attempt=0. Returns
        True if requeued, False if not found. Operators use this after fixing
        the underlying problem."""
        with self._lock, self._txn() as conn:
            r = conn.execute(
                "SELECT label, payload_json, enqueued_at FROM dead_letter WHERE id=?",
                (row_id,),
            ).fetchone()
            if r is None:
                return False
            now = _now()
            conn.execute(
                """
                INSERT INTO pending
                (id, label, payload_json, attempt, max_attempts,
                 next_attempt_at, last_error, enqueued_at, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, NULL, ?, ?)
                """,
                (row_id, r["label"], r["payload_json"], self.max_attempts, now, now, now),
            )
            conn.execute("DELETE FROM dead_letter WHERE id=?", (row_id,))
        logger.info(f"[RetryQueue] requeued dead-letter id={row_id}")
        return True

    def purge_dead_letter(self) -> int:
        with self._lock, self._txn() as conn:
            cur = conn.execute("DELETE FROM dead_letter")
            return cur.rowcount or 0
