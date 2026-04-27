"""Unit tests for QdrantVecDB's retry + unreachable classification helper.

These exercise _is_unavailable_error and _with_retry without a real Qdrant
server. The class is constructed via __new__ to skip __init__'s connection
attempt.
"""

from __future__ import annotations

import pytest

from memos.storage.exceptions import QdrantUnavailable
from memos.vec_dbs.qdrant import QdrantVecDB, _is_unavailable_error


class _FakeConnectionError(Exception):
    pass


_FakeConnectionError.__name__ = "ConnectionError"


class _FakeStatus500(Exception):
    def __init__(self, message="boom"):
        super().__init__(message)
        self.status_code = 500


class _FakeStatus404(Exception):
    def __init__(self, message="not found"):
        super().__init__(message)
        self.status_code = 404


def _make_db(retry_attempts: int = 3) -> QdrantVecDB:
    db = QdrantVecDB.__new__(QdrantVecDB)
    db._retry_attempts = retry_attempts
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────


class TestIsUnavailableError:
    def test_recognized_connection_error(self):
        assert _is_unavailable_error(_FakeConnectionError("connection reset"))

    def test_5xx_status_code(self):
        assert _is_unavailable_error(_FakeStatus500())

    def test_4xx_is_not_unavailable(self):
        # 404 / bad request → caller error, not retry-able
        assert not _is_unavailable_error(_FakeStatus404())

    def test_arbitrary_value_error_is_not_unavailable(self):
        assert not _is_unavailable_error(ValueError("bad input"))

    def test_walks_cause_chain(self):
        try:
            try:
                raise _FakeConnectionError("low")
            except Exception as inner:
                raise RuntimeError("wrap") from inner
        except RuntimeError as wrapped:
            assert _is_unavailable_error(wrapped)


# ─────────────────────────────────────────────────────────────────────────────
# _with_retry
# ─────────────────────────────────────────────────────────────────────────────


class TestWithRetry:
    def test_propagates_non_connection_errors_immediately(self):
        db = _make_db()
        n = {"calls": 0}

        def op():
            n["calls"] += 1
            raise ValueError("bad arg")

        with pytest.raises(ValueError):
            db._with_retry("test", op)
        assert n["calls"] == 1, "must NOT retry on non-connection error"

    def test_returns_value_on_first_success(self):
        db = _make_db()
        assert db._with_retry("test", lambda: 42) == 42

    def test_retries_then_succeeds(self):
        db = _make_db(retry_attempts=3)
        n = {"calls": 0}

        def op():
            n["calls"] += 1
            if n["calls"] < 3:
                raise _FakeConnectionError("transient")
            return "ok"

        assert db._with_retry("test", op) == "ok"
        assert n["calls"] == 3

    def test_raises_qdrant_unavailable_after_max_attempts(self):
        db = _make_db(retry_attempts=2)
        n = {"calls": 0}

        def op():
            n["calls"] += 1
            raise _FakeConnectionError("always down")

        with pytest.raises(QdrantUnavailable) as ei:
            db._with_retry("upsert", op)
        assert ei.value.dep_name == "qdrant"
        assert "upsert" in str(ei.value)
        assert n["calls"] == 2

    def test_5xx_triggers_retry(self):
        db = _make_db(retry_attempts=3)
        n = {"calls": 0}

        def op():
            n["calls"] += 1
            if n["calls"] < 2:
                raise _FakeStatus500()
            return "ok"

        assert db._with_retry("test", op) == "ok"
        assert n["calls"] == 2
