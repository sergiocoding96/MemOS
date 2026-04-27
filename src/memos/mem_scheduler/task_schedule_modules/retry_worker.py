"""
Retry-side handler that drains the durable retry queue and re-runs failed
extraction tasks against the live mem_cube registry.

Background (Bug 2 close-out): the storage agent shipped the durable enqueue
side of the retry queue (dispatcher.py:_create_task_wrapper), exposed
SchedulerDispatcher.start_retry_worker(handler), and explicitly deferred the
handler that rehydrates a mem_cube from its mem_cube_id and re-invokes the
original extraction. This module is that handler.

Wiring (v1):
    - The single mem_cube registry in v1 is `BaseScheduler._mem_cubes`
      (a dict[mem_cube_id -> BaseMemCube] shared by reference with
      MOSCore.mem_cubes via `self._mem_scheduler.mem_cubes = self.mem_cubes`
      in MOSCore.__init__). RetryWorker reads through `scheduler.mem_cubes`,
      not directly from MOSCore, so unit tests can supply a dict in place of
      a real MOSCore.
    - The original handlers are registered on the dispatcher
      (`dispatcher.handlers: dict[label, callable]`). On retry we look up
      `dispatcher.handlers[original_label]` and call the *raw* registered
      handler. We deliberately do NOT route through `execute_task` — that
      path goes back through the dispatcher's metric/wrapper which would
      re-enqueue the message on failure, producing a queue feedback loop.

Failure semantics (matches Bug 2 brief):
    - Cube no longer exists                    → RetryAbort("cube_no_longer_exists")
    - No registered handler for original label → RetryAbort("no_handler_for_<label>")
    - Dependency-class error during retry      → re-raise (queue retries with backoff)
    - Programming-class error during retry     → RetryAbort("programming_error: ...")

The classification reuses `SchedulerDispatcher._should_retry` so this module
and the original-call enqueue path agree on what "transient" means.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any

from memos.log import get_logger
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem
from memos.storage.retry_queue import RetryAbort


if TYPE_CHECKING:
    from memos.mem_scheduler.base_scheduler import BaseScheduler


logger = get_logger(__name__)


RETRY_LABEL_PREFIX = "retry::"
DISABLE_ENV_VAR = "MEMOS_RETRY_WORKER_DISABLED"


class RetryWorker:
    """Drains the dispatcher's retry queue against the v1 mem_cube registry.

    Lifecycle:
        worker = RetryWorker(scheduler)
        worker.start()          # idempotent; respects DISABLE_ENV_VAR
        ...
        worker.stop(timeout=5)  # best-effort daemon-thread join
    """

    def __init__(self, scheduler: BaseScheduler):
        self.scheduler = scheduler
        self._started = False

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start(self, *, thread_name: str = "memos-scheduler-retry") -> bool:
        """Spawn the queue's worker thread. Returns True if started, False
        if the disable env var is set or if the dispatcher has no retry
        queue wired (in which case there's nothing to drain)."""
        if os.getenv(DISABLE_ENV_VAR) == "1":
            logger.info(
                "[RetryWorker] %s=1; not starting retry worker", DISABLE_ENV_VAR
            )
            return False
        dispatcher = getattr(self.scheduler, "dispatcher", None)
        if dispatcher is None:
            logger.warning("[RetryWorker] scheduler has no dispatcher; skipping start")
            return False
        if getattr(dispatcher, "retry_queue", None) is None:
            logger.info("[RetryWorker] dispatcher.retry_queue is None; skipping start")
            return False
        dispatcher.start_retry_worker(self.handle, thread_name=thread_name)
        self._started = True
        logger.info("[RetryWorker] started (thread=%s)", thread_name)
        return True

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        dispatcher = getattr(self.scheduler, "dispatcher", None)
        if dispatcher is None:
            return
        dispatcher.stop_retry_worker(timeout=timeout)
        self._started = False
        logger.info("[RetryWorker] stopped")

    # ──────────────────────────────────────────────────────────────────────
    # Handler
    # ──────────────────────────────────────────────────────────────────────

    def handle(self, label: str, payload: dict[str, Any]) -> None:
        """Queue handler signature: handler(label, payload) -> None.

        Returns normally on success (queue marks the row complete and deletes
        it). Raises RetryAbort to dead-letter immediately, or any other
        exception to trigger normal retry+backoff via the queue.
        """
        original_label = (
            label[len(RETRY_LABEL_PREFIX):] if label.startswith(RETRY_LABEL_PREFIX) else label
        )
        mem_cube_id = payload.get("mem_cube_id")

        cube = self._lookup_cube(mem_cube_id)
        if cube is None:
            raise RetryAbort(f"cube_no_longer_exists: mem_cube_id={mem_cube_id!r}")

        original_handler = self._lookup_handler(original_label)
        if original_handler is None:
            raise RetryAbort(f"no_handler_for_{original_label}")

        msg = self._reconstruct_message(payload, original_label=original_label, cube=cube)

        try:
            original_handler([msg])
        except RetryAbort:
            raise
        except Exception as e:
            if self._is_dependency_error(e):
                # Let the queue's normal retry+backoff path handle it.
                raise
            # Programming-class error — fail loud, don't retry forever.
            raise RetryAbort(
                f"programming_error: {type(e).__name__}: {e}"
            ) from e

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _lookup_cube(self, mem_cube_id: str | None):
        if not mem_cube_id:
            return None
        registry = getattr(self.scheduler, "mem_cubes", None)
        if registry is None:
            return None
        try:
            return registry.get(mem_cube_id)
        except Exception:
            # registry might be a non-dict thread-safe wrapper; fall back to []
            try:
                return registry[mem_cube_id] if mem_cube_id in registry else None
            except Exception:
                return None

    def _lookup_handler(self, original_label: str):
        dispatcher = getattr(self.scheduler, "dispatcher", None)
        if dispatcher is None:
            return None
        handlers = getattr(dispatcher, "handlers", None)
        if not handlers:
            return None
        return handlers.get(original_label)

    @staticmethod
    def _reconstruct_message(
        payload: dict[str, Any], *, original_label: str, cube
    ) -> ScheduleMessageItem:
        """Rebuild a ScheduleMessageItem from the durable payload. Live
        in-process objects (mem_cube) are re-attached from the registry
        lookup, not from the payload."""
        return ScheduleMessageItem(
            item_id=payload.get("item_id") or "",
            user_id=payload["user_id"],
            mem_cube_id=payload["mem_cube_id"],
            mem_cube=cube,
            session_id=payload.get("session_id", "") or "",
            label=original_label,
            content=payload.get("content", "") or "",
            user_name=payload.get("user_name", "") or "",
            task_id=payload.get("task_id"),
            info=payload.get("info"),
            chat_history=payload.get("chat_history"),
            trace_id=payload.get("trace_id") or None,
        )

    @staticmethod
    def _is_dependency_error(exc: BaseException) -> bool:
        """Mirror SchedulerDispatcher._should_retry semantics: a transient
        dependency outage worth retrying. Implemented here (not delegated)
        because dispatcher's version is bound and walks the cause chain via
        `self.` recursion. The classification rules must stay aligned with
        SchedulerDispatcher._should_retry — when one is updated, update both.
        """
        from memos.storage.exceptions import DependencyUnavailable

        if isinstance(exc, DependencyUnavailable):
            return True
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        if cause is not None and cause is not exc:
            return RetryWorker._is_dependency_error(cause)
        if type(exc).__module__.startswith("neo4j") and type(exc).__name__ in {
            "ServiceUnavailable",
            "RoutingServiceUnavailable",
            "WriteServiceUnavailable",
            "ReadServiceUnavailable",
            "DatabaseUnavailable",
            "ConnectionAcquisitionTimeoutError",
            "SessionExpired",
        }:
            return True
        return False
