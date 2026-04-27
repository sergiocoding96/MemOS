"""Auth tests for the Admin Router (F-02 follow-up).

The admin key compare must be constant-time so an attacker cannot use
response-timing to learn the admin key character-by-character. We can't
*directly* assert "this is constant time" from a unit test (that would
need a statistical timing harness), but we can assert the function
delegates to ``hmac.compare_digest`` on the secret-equality decision —
which is the right shape regardless of CPU.

We also pin the surrounding behavior so a future refactor cannot
silently regress the const-time check or the 401/503 contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# Bind the real ``compare_digest`` reference *before* any test patches the
# `admin_router.hmac` namespace. We delegate the spy to this captured
# function so the spy never recurses into its own patch.
from hmac import compare_digest as _real_compare_digest

import pytest
from fastapi import HTTPException

from memos.api.routers import admin_router


@pytest.fixture
def patched_admin_key(monkeypatch):
    """Force a known admin key into the module-level constant.

    `_ADMIN_KEY` is read once at import time from the env, so we patch
    the attribute on the imported module rather than the env var.
    """
    monkeypatch.setattr(admin_router, "_ADMIN_KEY", "secret-admin-key-12345")
    return "secret-admin-key-12345"


def _make_request(authorization: str | None) -> MagicMock:
    """Mock just enough of starlette.Request for `_require_admin`."""
    req = MagicMock()
    headers = {} if authorization is None else {"Authorization": authorization}
    # `request.headers.get(name, default)` is what the code uses.
    req.headers.get = lambda name, default="": headers.get(name, default)
    return req


def test_require_admin_accepts_correct_key(patched_admin_key):
    """The happy path returns None (no exception) on a valid Bearer token."""
    req = _make_request(f"Bearer {patched_admin_key}")
    assert admin_router._require_admin(req) is None


def test_require_admin_rejects_wrong_key(patched_admin_key):
    req = _make_request("Bearer not-the-admin-key")
    with pytest.raises(HTTPException) as exc:
        admin_router._require_admin(req)
    assert exc.value.status_code == 401


def test_require_admin_rejects_missing_header(patched_admin_key):
    req = _make_request(None)
    with pytest.raises(HTTPException) as exc:
        admin_router._require_admin(req)
    assert exc.value.status_code == 401


def test_require_admin_rejects_wrong_scheme(patched_admin_key):
    req = _make_request(f"Basic {patched_admin_key}")
    with pytest.raises(HTTPException) as exc:
        admin_router._require_admin(req)
    assert exc.value.status_code == 401


def test_require_admin_returns_503_when_unconfigured(monkeypatch):
    monkeypatch.setattr(admin_router, "_ADMIN_KEY", "")
    req = _make_request("Bearer anything")
    with pytest.raises(HTTPException) as exc:
        admin_router._require_admin(req)
    assert exc.value.status_code == 503


def test_require_admin_uses_constant_time_compare(patched_admin_key):
    """The key-equality decision must go through ``hmac.compare_digest``.

    This is the F-02 fix's invariant: if a future refactor swaps
    `hmac.compare_digest(...)` for `==`, this test must fail. We patch
    the symbol the module imported (``admin_router.hmac.compare_digest``),
    record what arguments it sees, and verify both that it was called
    and that the comparison was on bytes (compare_digest rejects str
    pairs vs. bytes pairs in a way that would mask a bug).
    """
    calls: list[tuple] = []

    def _spy(a, b):
        calls.append((a, b))
        return _real_compare_digest(a, b)

    with patch.object(admin_router.hmac, "compare_digest", side_effect=_spy):
        req = _make_request(f"Bearer {patched_admin_key}")
        admin_router._require_admin(req)

    assert calls, "_require_admin did not delegate to hmac.compare_digest"
    a, b = calls[-1]
    # Both args must be bytes; compare_digest handles bytes uniformly,
    # while str-vs-str works on CPython but is a stricter invariant to
    # encode to bytes explicitly (and avoids any locale/Unicode footgun).
    assert isinstance(a, bytes) and isinstance(b, bytes)
    assert b == patched_admin_key.encode()


def test_require_admin_rejects_via_compare_digest_on_close_match(patched_admin_key):
    """A near-match (one char off) must still go through compare_digest
    and still be rejected with 401 — proves we didn't add a fast-path
    that bypasses the constant-time path on long keys.
    """
    near = patched_admin_key[:-1] + "X"  # change last char
    calls: list[tuple] = []

    def _spy(a, b):
        calls.append((a, b))
        return _real_compare_digest(a, b)

    with patch.object(admin_router.hmac, "compare_digest", side_effect=_spy):
        req = _make_request(f"Bearer {near}")
        with pytest.raises(HTTPException) as exc:
            admin_router._require_admin(req)
        assert exc.value.status_code == 401
    assert calls, "near-match request did not reach hmac.compare_digest"
