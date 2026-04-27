"""
Typed exceptions for storage-layer dependency outages.

When a downstream storage dependency (Qdrant, Neo4j, LLM provider) is
unreachable during a request, the driver should raise a subclass of
`DependencyUnavailable`. The API exception handler maps these to HTTP 503
with a body that names the offending dependency, replacing the prior
behavior of returning HTTP 200 with a silently-lost write.
"""

from __future__ import annotations


class DependencyUnavailable(Exception):
    """Base class for storage-layer dependency outages.

    Attributes:
        dep_name: Short name of the dependency, e.g. "qdrant", "neo4j", "llm".
                  Used in the HTTP 503 body and in /health/deps responses.
    """

    dep_name: str = "unknown"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover — convenience
        base = super().__str__()
        if self.cause is not None:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base


class QdrantUnavailable(DependencyUnavailable):
    dep_name = "qdrant"


class Neo4jUnavailable(DependencyUnavailable):
    dep_name = "neo4j"


class LLMUnavailable(DependencyUnavailable):
    dep_name = "llm"
