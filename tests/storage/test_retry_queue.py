"""Unit tests for the SQLite-backed durable retry queue."""

from __future__ import annotations

import os
import sqlite3
import threading
import time

from pathlib import Path

import pytest

from memos.storage.retry_queue import RetryQueue, _backoff_seconds


@pytest.fixture
def tmp_queue(tmp_path: Path) -> RetryQueue:
    db = tmp_path / "retry.sqlite"
    return RetryQueue(
        db_path=str(db),
        max_attempts=3,
        poll_interval_s=0.05,
        backoff_initial_s=0.05,
        backoff_cap_s=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backoff math
# ─────────────────────────────────────────────────────────────────────────────


class TestBackoff:
    def test_attempt_0_or_negative_is_zero(self):
        assert _backoff_seconds(0) == 0.0
        assert _backoff_seconds(-5) == 0.0

    def test_exponential_growth(self):
        assert _backoff_seconds(1, initial=1.0, cap=60.0) == 1.0
        assert _backoff_seconds(2, initial=1.0, cap=60.0) == 2.0
        assert _backoff_seconds(3, initial=1.0, cap=60.0) == 4.0
        assert _backoff_seconds(4, initial=1.0, cap=60.0) == 8.0

    def test_caps_at_cap(self):
        assert _backoff_seconds(20, initial=1.0, cap=60.0) == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Schema + enqueue/dequeue happy path
# ─────────────────────────────────────────────────────────────────────────────


class TestEnqueueDequeue:
    def test_schema_created(self, tmp_queue: RetryQueue):
        # Probe the SQLite file directly
        conn = sqlite3.connect(tmp_queue.db_path)
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert {"pending", "dead_letter"}.issubset(tables)

    def test_enqueue_returns_id_and_persists(self, tmp_queue: RetryQueue):
        rid = tmp_queue.enqueue("test::label", {"k": "v"})
        assert isinstance(rid, str) and len(rid) > 8
        assert tmp_queue.pending_count() == 1

    def test_drain_once_processes_due_rows(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("test::label", {"i": 1})
        tmp_queue.enqueue("test::label", {"i": 2})

        seen: list[dict] = []

        def handler(label, payload):
            seen.append(payload)

        n = tmp_queue.drain_once(handler)
        assert n == 2
        assert {p["i"] for p in seen} == {1, 2}
        assert tmp_queue.pending_count() == 0

    def test_drain_skips_not_yet_due(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("a", {}, delay_s=10.0)
        n = tmp_queue.drain_once(lambda l, p: None)
        assert n == 0
        assert tmp_queue.pending_count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Retry semantics
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrySemantics:
    def test_failure_reschedules_with_backoff(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("fail-once", {"x": 1})

        attempts = {"n": 0}

        def handler(label, payload):
            attempts["n"] += 1
            raise RuntimeError("boom")

        # First drain → handler raises → row is rescheduled, NOT dead-lettered
        tmp_queue.drain_once(handler)
        assert attempts["n"] == 1
        assert tmp_queue.pending_count() == 1
        assert tmp_queue.dead_letter_count() == 0

        # Without sleeping, the row is not yet due (backoff_initial=0.05s)
        assert tmp_queue.drain_once(handler) == 0
        assert attempts["n"] == 1

        # Sleep past backoff → drain picks it up again
        time.sleep(0.2)
        tmp_queue.drain_once(handler)
        assert attempts["n"] == 2

    def test_max_attempts_moves_to_dead_letter(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("always-fails", {"x": 1})  # max_attempts=3

        def handler(label, payload):
            raise RuntimeError("boom")

        for _ in range(5):
            tmp_queue.drain_once(handler)
            time.sleep(0.6)  # past backoff cap

        assert tmp_queue.pending_count() == 0
        assert tmp_queue.dead_letter_count() == 1

        rows = tmp_queue.list_dead_letter()
        assert len(rows) == 1
        assert rows[0]["attempts"] == 3
        assert rows[0]["last_error"].startswith("RuntimeError")

    def test_success_clears_row(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("ok", {"x": 1})
        tmp_queue.drain_once(lambda l, p: None)
        assert tmp_queue.pending_count() == 0
        assert tmp_queue.dead_letter_count() == 0

    def test_success_after_failure_clears_row(self, tmp_queue: RetryQueue):
        tmp_queue.enqueue("ok-eventually", {"x": 1})
        n = {"calls": 0}

        def handler(label, payload):
            n["calls"] += 1
            if n["calls"] < 2:
                raise RuntimeError("transient")

        tmp_queue.drain_once(handler)
        assert tmp_queue.pending_count() == 1
        time.sleep(0.2)
        tmp_queue.drain_once(handler)
        assert tmp_queue.pending_count() == 0
        assert tmp_queue.dead_letter_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────


class TestWorker:
    def test_worker_processes_in_background(self, tmp_queue: RetryQueue):
        seen = threading.Event()

        def handler(label, payload):
            if payload.get("trigger"):
                seen.set()

        tmp_queue.start_worker(handler)
        try:
            tmp_queue.enqueue("bg", {"trigger": True})
            assert seen.wait(timeout=2.0), "worker did not process row in 2s"
        finally:
            tmp_queue.stop_worker()

    def test_worker_survives_handler_exception(self, tmp_queue: RetryQueue):
        # Defensive: a buggy handler that raises must NOT kill the worker.
        # After the failure the row should be rescheduled and a follow-up
        # success should still drain.
        events = {"ok": threading.Event(), "fail_seen": 0}

        def handler(label, payload):
            if payload.get("kind") == "fail":
                events["fail_seen"] += 1
                raise RuntimeError("nope")
            events["ok"].set()

        tmp_queue.start_worker(handler)
        try:
            tmp_queue.enqueue("first", {"kind": "fail"})
            tmp_queue.enqueue("second", {"kind": "ok"}, delay_s=0.1)
            assert events["ok"].wait(timeout=3.0)
            assert events["fail_seen"] >= 1
        finally:
            tmp_queue.stop_worker()


# ─────────────────────────────────────────────────────────────────────────────
# Persistence across instance restarts
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_pending_survives_process_restart(self, tmp_path: Path):
        db = str(tmp_path / "p.sqlite")
        q1 = RetryQueue(db_path=db)
        rid = q1.enqueue("survives", {"x": 1})
        assert q1.pending_count() == 1

        # Simulate process restart by constructing a fresh instance
        q2 = RetryQueue(db_path=db)
        assert q2.pending_count() == 1

        # And the row still has the id we expect
        seen = []
        q2.drain_once(lambda l, p: seen.append((l, p)))
        assert seen == [("survives", {"x": 1})]

    def test_dead_letter_survives_process_restart(self, tmp_path: Path):
        db = str(tmp_path / "p.sqlite")
        q1 = RetryQueue(db_path=db, max_attempts=1)
        q1.enqueue("dead", {"x": 1})
        q1.drain_once(lambda l, p: (_ for _ in ()).throw(RuntimeError("boom")))
        assert q1.dead_letter_count() == 1

        q2 = RetryQueue(db_path=db)
        assert q2.dead_letter_count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Maintenance: requeue dead-letter
# ─────────────────────────────────────────────────────────────────────────────


class TestRequeueDeadLetter:
    def test_requeue_brings_row_back(self, tmp_queue: RetryQueue):
        rid = tmp_queue.enqueue("d", {"x": 1})
        # force into dead letter
        for _ in range(5):
            tmp_queue.drain_once(lambda l, p: (_ for _ in ()).throw(RuntimeError("x")))
            time.sleep(0.6)
        assert tmp_queue.dead_letter_count() == 1
        rows = tmp_queue.list_dead_letter()

        ok = tmp_queue.requeue_dead_letter(rows[0]["id"])
        assert ok is True
        assert tmp_queue.dead_letter_count() == 0
        assert tmp_queue.pending_count() == 1

    def test_requeue_unknown_id_returns_false(self, tmp_queue: RetryQueue):
        assert tmp_queue.requeue_dead_letter("does-not-exist") is False
