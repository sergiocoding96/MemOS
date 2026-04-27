"""Test that SchedulerDispatcher enqueues failed messages on the retry
queue when (and only when) the failure looks like a dependency outage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memos.mem_scheduler.task_schedule_modules.dispatcher import SchedulerDispatcher
from memos.storage.exceptions import QdrantUnavailable
from memos.storage.retry_queue import RetryQueue


def _make_dispatcher(retry_queue: RetryQueue) -> SchedulerDispatcher:
    """Construct via __new__ to skip the heavy __init__ (thread pool, etc.)."""
    d = SchedulerDispatcher.__new__(SchedulerDispatcher)
    d.retry_queue = retry_queue
    return d


def _fake_msg(label: str = "MEM_READ", item_id: str = "i1"):
    m = MagicMock()
    m.item_id = item_id
    m.user_id = "u1"
    m.mem_cube_id = "cube-1"
    m.session_id = "sess"
    m.label = label
    m.content = '["mid-1","mid-2"]'
    m.user_name = "alice"
    m.task_id = "tsk-1"
    m.info = {"k": "v"}
    m.chat_history = None
    m.trace_id = "tid"
    m.timestamp = None
    return m


# ─────────────────────────────────────────────────────────────────────────────
# _should_retry: dependency-class only
# ─────────────────────────────────────────────────────────────────────────────


class TestShouldRetry:
    def test_qdrant_unavailable_yes(self):
        d = _make_dispatcher(MagicMock())
        assert d._should_retry(QdrantUnavailable("nope")) is True

    def test_value_error_no(self):
        d = _make_dispatcher(MagicMock())
        assert d._should_retry(ValueError("bad")) is False

    def test_neo4j_service_unavailable_yes(self):
        class FakeServiceUnavailable(Exception):
            pass

        FakeServiceUnavailable.__module__ = "neo4j.exceptions"
        FakeServiceUnavailable.__name__ = "ServiceUnavailable"
        d = _make_dispatcher(MagicMock())
        assert d._should_retry(FakeServiceUnavailable("conn")) is True

    def test_walks_cause_chain(self):
        d = _make_dispatcher(MagicMock())
        try:
            try:
                raise QdrantUnavailable("inner")
            except Exception as inner:
                raise RuntimeError("wrap") from inner
        except RuntimeError as wrapped:
            assert d._should_retry(wrapped) is True


# ─────────────────────────────────────────────────────────────────────────────
# _serialize_message_for_retry: durable subset only
# ─────────────────────────────────────────────────────────────────────────────


class TestSerializeForRetry:
    def test_serializes_durable_fields(self):
        msg = _fake_msg(label="MEM_READ", item_id="i1")
        payload = SchedulerDispatcher._serialize_message_for_retry(
            msg, original_error=QdrantUnavailable("simulated")
        )
        assert payload["item_id"] == "i1"
        assert payload["user_id"] == "u1"
        assert payload["mem_cube_id"] == "cube-1"
        assert payload["label"] == "MEM_READ"
        assert payload["content"] == '["mid-1","mid-2"]'
        assert payload["task_id"] == "tsk-1"
        # mem_cube live object is NOT serialized
        assert "mem_cube" not in payload
        # Original error metadata is captured for forensics
        assert payload["_retry_meta"]["original_error_type"] == "QdrantUnavailable"
        assert "simulated" in payload["_retry_meta"]["original_error_msg"]


# ─────────────────────────────────────────────────────────────────────────────
# Real RetryQueue integration: dispatcher._should_retry + enqueue
# ─────────────────────────────────────────────────────────────────────────────


class TestEnqueueOnFailure:
    """Exercise the snippet that runs inside the wrapped_handler's except
    block. We don't run wrapped_handler itself (it pulls in the full
    scheduler graph). We test the enqueue behaviour in isolation."""

    def test_dep_failure_enqueues(self, tmp_path: Path):
        q = RetryQueue(db_path=str(tmp_path / "q.sqlite"), max_attempts=3)
        d = _make_dispatcher(q)
        msg = _fake_msg()
        err = QdrantUnavailable("down")

        # The relevant branch from _create_task_wrapper, copied verbatim:
        if d.retry_queue is not None and d._should_retry(err):
            payload = d._serialize_message_for_retry(msg, original_error=err)
            d.retry_queue.enqueue(label=f"retry::{msg.label}", payload=payload)

        assert q.pending_count() == 1
        # Drain the row so we can confirm payload shape
        seen = []
        q.drain_once(lambda l, p: seen.append((l, p)))
        assert len(seen) == 1
        label, payload = seen[0]
        assert label == "retry::MEM_READ"
        assert payload["item_id"] == "i1"
        assert payload["mem_cube_id"] == "cube-1"

    def test_programming_error_does_not_enqueue(self, tmp_path: Path):
        q = RetryQueue(db_path=str(tmp_path / "q.sqlite"), max_attempts=3)
        d = _make_dispatcher(q)
        msg = _fake_msg()
        err = ValueError("bad input")

        if d.retry_queue is not None and d._should_retry(err):
            d.retry_queue.enqueue(
                label=f"retry::{msg.label}",
                payload=d._serialize_message_for_retry(msg, original_error=err),
            )

        assert q.pending_count() == 0  # programming errors are NOT queued
