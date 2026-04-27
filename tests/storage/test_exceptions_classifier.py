"""Unit tests for the API-side dependency-error classifier."""

from __future__ import annotations

from memos.api.exceptions import _classify_dependency_error
from memos.storage.exceptions import (
    DependencyUnavailable,
    Neo4jUnavailable,
    QdrantUnavailable,
)


# Synthetic neo4j-shaped exception so we don't need the real driver installed
class _FakeNeo4jServiceUnavailable(Exception):
    pass


_FakeNeo4jServiceUnavailable.__module__ = "neo4j.exceptions"
_FakeNeo4jServiceUnavailable.__name__ = "ServiceUnavailable"


class TestClassifier:
    def test_dependency_unavailable_passes_through(self):
        exc = QdrantUnavailable("nope")
        out = _classify_dependency_error(exc)
        assert out is exc
        assert out.dep_name == "qdrant"

    def test_neo4j_service_unavailable_classified(self):
        out = _classify_dependency_error(_FakeNeo4jServiceUnavailable("conn refused"))
        assert isinstance(out, Neo4jUnavailable)
        assert out.dep_name == "neo4j"
        assert "conn refused" in str(out)

    def test_unrelated_error_returns_none(self):
        out = _classify_dependency_error(ValueError("just a value error"))
        assert out is None

    def test_walks_cause_chain(self):
        try:
            try:
                raise _FakeNeo4jServiceUnavailable("low-level")
            except Exception as inner:
                raise RuntimeError("wrapper") from inner
        except RuntimeError as wrapped:
            out = _classify_dependency_error(wrapped)
            assert isinstance(out, Neo4jUnavailable)


class TestExceptionShape:
    def test_dep_unavailable_has_dep_name(self):
        for cls, name in [
            (QdrantUnavailable, "qdrant"),
            (Neo4jUnavailable, "neo4j"),
        ]:
            exc = cls("x")
            assert isinstance(exc, DependencyUnavailable)
            assert exc.dep_name == name

    def test_cause_attached(self):
        cause = ValueError("inner")
        exc = QdrantUnavailable("outer", cause=cause)
        assert exc.cause is cause
        assert "ValueError" in str(exc)
