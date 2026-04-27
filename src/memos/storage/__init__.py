"""
memos.storage — write-path resilience.

Pieces:
- exceptions.DependencyUnavailable: typed exceptions raised by storage
  drivers (Qdrant, Neo4j, LLM) when a downstream is unreachable. The API
  layer maps these to HTTP 503 instead of returning a silently-lost 200.
- retry_queue.RetryQueue: SQLite-backed durable at-least-once queue with
  exponential backoff and a dead-letter table. Used by the scheduler to
  replace fire-and-forget extraction tasks (Bug 2 in the 2026-04-26
  storage audit).
- dependency_health.DependencyHealth: probes for Qdrant, Neo4j, and the
  configured LLM provider. Backs the /health and /health/deps endpoints.
"""

from memos.storage.dependency_health import DepStatus, DependencyHealth
from memos.storage.exceptions import (
    DependencyUnavailable,
    LLMUnavailable,
    Neo4jUnavailable,
    QdrantUnavailable,
)
from memos.storage.retry_queue import DEFAULT_QUEUE_PATH, RetryAbort, RetryQueue


__all__ = [
    "DEFAULT_QUEUE_PATH",
    "DepStatus",
    "DependencyHealth",
    "DependencyUnavailable",
    "LLMUnavailable",
    "Neo4jUnavailable",
    "QdrantUnavailable",
    "RetryAbort",
    "RetryQueue",
]
