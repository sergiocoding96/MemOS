"""Tests for the retry-side handler that closes the at-least-once contract.

Background: the dispatcher already enqueues failed messages to a durable
SQLite queue (see test_dispatcher_retry_enqueue.py). RetryWorker is the
handler the queue calls back when a row comes due — it rehydrates the
mem_cube from the live registry and re-runs the original extraction.

These tests use the queue's `drain_once` API rather than starting a real
background thread; that keeps them deterministic and fast. The thread-start
path is exercised separately (test_worker_disable_env_skips_start)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from memos.mem_scheduler.task_schedule_modules.retry_worker import (
    DISABLE_ENV_VAR,
    RETRY_LABEL_PREFIX,
    RetryWorker,
)
from memos.storage.exceptions import QdrantUnavailable
from memos.storage.retry_queue import RetryQueue


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_LABEL = "MEM_READ"
RETRY_LABEL = f"{RETRY_LABEL_PREFIX}{ORIGINAL_LABEL}"
CUBE_ID = "cube-1"
USER_ID = "u1"


def _payload(*, mem_cube_id: str = CUBE_ID, content: str = "hello") -> dict[str, Any]:
    """Mirror the durable subset SchedulerDispatcher._serialize_message_for_retry
    produces. Keep aligned with that method's keys."""
    return {
        "item_id": "item-1",
        "user_id": USER_ID,
        "mem_cube_id": mem_cube_id,
        "session_id": "sess-1",
        "label": ORIGINAL_LABEL,
        "content": content,
        "user_name": "alice",
        "task_id": "task-1",
        "info": {"k": "v"},
        "chat_history": None,
        "trace_id": "trace-1",
        "timestamp": None,
        "_retry_meta": {
            "original_error_type": "QdrantUnavailable",
            "original_error_msg": "simulated",
        },
    }


@pytest.fixture
def fast_queue(tmp_path: Path) -> RetryQueue:
    return RetryQueue(
        db_path=str(tmp_path / "retry.sqlite"),
        max_attempts=3,
        poll_interval_s=0.01,
        backoff_initial_s=0.0,
        backoff_cap_s=0.0,
    )


def _make_scheduler(
    *,
    queue: RetryQueue,
    cube=object(),
    cube_id: str = CUBE_ID,
    handler=None,
):
    """Build a minimal duck-typed scheduler for RetryWorker.

    Attributes RetryWorker reads:
      - scheduler.mem_cubes        (dict[mem_cube_id -> cube])
      - scheduler.dispatcher.handlers       (dict[label -> callable])
      - scheduler.dispatcher.retry_queue    (RetryQueue)
      - scheduler.dispatcher.start_retry_worker / stop_retry_worker
    """
    handlers: dict[str, Any] = {}
    if handler is not None:
        handlers[ORIGINAL_LABEL] = handler

    dispatcher = SimpleNamespace(
        handlers=handlers,
        retry_queue=queue,
        start_retry_worker=lambda h, *, thread_name="t": queue.start_worker(
            h, thread_name=thread_name
        ),
        stop_retry_worker=lambda timeout=5.0: queue.stop_worker(timeout=timeout),
    )
    cubes = {cube_id: cube} if cube is not None else {}
    return SimpleNamespace(mem_cubes=cubes, dispatcher=dispatcher)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryWorker:
    def test_worker_drains_queue_on_success(self, fast_queue: RetryQueue):
        seen = []

        def passing_handler(messages):
            assert len(messages) == 1
            assert messages[0].mem_cube_id == CUBE_ID
            seen.append(messages[0].content)

        scheduler = _make_scheduler(queue=fast_queue, handler=passing_handler)
        worker = RetryWorker(scheduler)
        fast_queue.enqueue(label=RETRY_LABEL, payload=_payload())

        processed = fast_queue.drain_once(worker.handle)

        assert processed == 1
        assert seen == ["hello"]
        assert fast_queue.pending_count() == 0
        assert fast_queue.dead_letter_count() == 0

    def test_worker_retries_dependency_failure(self, fast_queue: RetryQueue):
        """Fails 3x with QdrantUnavailable, then succeeds. The queue's
        max_attempts is 3, so attempt #3 (1-indexed) is the last allowed.
        We drive 3 drain_once calls: first two fail (reschedule), third
        succeeds (delete row)."""
        attempts: list[int] = []

        def flaky_handler(messages):
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise QdrantUnavailable("transient")
            # 3rd attempt succeeds

        scheduler = _make_scheduler(queue=fast_queue, handler=flaky_handler)
        worker = RetryWorker(scheduler)
        fast_queue.enqueue(label=RETRY_LABEL, payload=_payload())

        # Three drain_once cycles — backoff is 0s so all are immediately due.
        fast_queue.drain_once(worker.handle)
        fast_queue.drain_once(worker.handle)
        fast_queue.drain_once(worker.handle)

        assert len(attempts) == 3
        assert fast_queue.pending_count() == 0
        assert fast_queue.dead_letter_count() == 0  # final attempt succeeded

    def test_worker_dead_letters_after_max_attempts(self, fast_queue: RetryQueue):
        """A handler that always raises a dependency error should retry up
        to max_attempts then dead-letter (the queue's existing logic, with
        RetryWorker re-raising dep errors so the queue can retry)."""
        call_count = {"n": 0}

        def always_dep_fails(messages):
            call_count["n"] += 1
            raise QdrantUnavailable("still down")

        scheduler = _make_scheduler(queue=fast_queue, handler=always_dep_fails)
        worker = RetryWorker(scheduler)
        fast_queue.enqueue(label=RETRY_LABEL, payload=_payload())

        # Drain repeatedly until pending is empty (max_attempts=3).
        for _ in range(5):
            if fast_queue.pending_count() == 0:
                break
            fast_queue.drain_once(worker.handle)

        assert fast_queue.pending_count() == 0
        assert fast_queue.dead_letter_count() == 1
        assert call_count["n"] == 3
        dead = fast_queue.list_dead_letter()
        assert dead[0]["attempts"] == 3
        assert "QdrantUnavailable" in dead[0]["last_error"]

    def test_worker_dead_letters_programming_error(self, fast_queue: RetryQueue):
        """A non-retryable (programming-class) error should go straight to
        dead-letter on the first attempt — no retry loop on bugs."""

        def buggy_handler(messages):
            raise ValueError("forgot to handle empty content")

        scheduler = _make_scheduler(queue=fast_queue, handler=buggy_handler)
        worker = RetryWorker(scheduler)
        fast_queue.enqueue(label=RETRY_LABEL, payload=_payload())

        fast_queue.drain_once(worker.handle)

        assert fast_queue.pending_count() == 0
        assert fast_queue.dead_letter_count() == 1
        dead = fast_queue.list_dead_letter()
        assert dead[0]["attempts"] == 1  # first try, no retry
        assert "RetryAbort" in dead[0]["last_error"]
        assert "programming_error" in dead[0]["last_error"]
        assert "ValueError" in dead[0]["last_error"]

    def test_worker_handles_missing_cube(self, fast_queue: RetryQueue):
        """Cube was deleted between enqueue and retry — payload's
        mem_cube_id is no longer in the registry. Worker dead-letters
        with a 'cube_no_longer_exists' reason rather than retrying forever."""

        def should_not_be_called(messages):
            raise AssertionError("handler must not run when cube is missing")

        scheduler = _make_scheduler(
            queue=fast_queue, cube=None, cube_id="other", handler=should_not_be_called
        )
        worker = RetryWorker(scheduler)
        # Payload references CUBE_ID; registry only has 'other'.
        fast_queue.enqueue(label=RETRY_LABEL, payload=_payload(mem_cube_id=CUBE_ID))

        fast_queue.drain_once(worker.handle)

        assert fast_queue.pending_count() == 0
        assert fast_queue.dead_letter_count() == 1
        dead = fast_queue.list_dead_letter()
        assert dead[0]["attempts"] == 1
        assert "cube_no_longer_exists" in dead[0]["last_error"]
        assert dead[0]["payload"]["mem_cube_id"] == CUBE_ID  # forensic preservation

    def test_worker_disable_env_skips_start(
        self, fast_queue: RetryQueue, monkeypatch: pytest.MonkeyPatch
    ):
        """With MEMOS_RETRY_WORKER_DISABLED=1 set, RetryWorker.start() is a
        no-op: returns False, no thread spawned, queue never sees a worker."""
        monkeypatch.setenv(DISABLE_ENV_VAR, "1")

        scheduler = _make_scheduler(queue=fast_queue, handler=lambda m: None)
        # Spy on start_retry_worker to confirm it is never called.
        spy = MagicMock(side_effect=scheduler.dispatcher.start_retry_worker)
        scheduler.dispatcher.start_retry_worker = spy

        worker = RetryWorker(scheduler)
        result = worker.start()

        assert result is False
        assert spy.call_count == 0
        # And stop is also a no-op without a prior start.
        worker.stop()
