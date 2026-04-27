"""Tests for the BCrypt min-cost guard (F-03 from the 2026-04-26
zero-knowledge audit).

The guard lives in ``AgentAuthMiddleware._load_config`` and rejects any
hash with cost factor below ``MIN_BCRYPT_COST`` (currently 10). A
weakened hash is still auth-effective on its own, but undermines the
per-request CPU floor the rate limiter assumes — an attacker who can
plant a weak hash effectively raises the brute-force ceiling by 100×.

These tests pin three invariants:

1. **Default hashes load.** Real production agents use rounds=12 — they
   must continue to authenticate.
2. **Weak hashes are rejected, not loaded.** A rounds=4 hash for the
   same agent must NOT end up in the agent registry, even if everything
   else in the entry is valid.
3. **Unparseable hashes are rejected, not silently loaded.** A
   corrupted ``key_hash`` string must be skipped and logged at ERROR.
4. **The cost-parser itself** handles the obvious edge cases without
   crashing (empty string, wrong prefix, non-string).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import bcrypt
import pytest

from memos.api.middleware.agent_auth import (
    MIN_BCRYPT_COST,
    AgentAuthMiddleware,
    _parse_bcrypt_cost,
)


# ---------------------------------------------------------------------------
# _parse_bcrypt_cost — the cheap pre-validation helper
# ---------------------------------------------------------------------------


def test_parse_cost_default_hash_is_12():
    h = bcrypt.hashpw(b"key", bcrypt.gensalt(rounds=12)).decode()
    assert _parse_bcrypt_cost(h) == 12


def test_parse_cost_weak_hash_is_4():
    h = bcrypt.hashpw(b"key", bcrypt.gensalt(rounds=4)).decode()
    assert _parse_bcrypt_cost(h) == 4


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "$2b$XX$abc",  # non-numeric cost
        "not-a-hash",
        "$1$foo$bar",  # md5 crypt prefix, not bcrypt
        "$2",  # truncated
        None,  # type: ignore[list-item]
        b"$2b$12$abc",  # bytes, not str
        12,  # int
    ],
)
def test_parse_cost_returns_none_on_garbage(bad):
    assert _parse_bcrypt_cost(bad) is None


# ---------------------------------------------------------------------------
# _load_config — full integration through the middleware constructor
# ---------------------------------------------------------------------------


def _write_v2_config(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"version": 2, "agents": entries}))


def _hashed_entry(raw_key: str, user_id: str, *, rounds: int = 12) -> dict:
    return {
        "key_hash": bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt(rounds=rounds)).decode(),
        "key_prefix": raw_key[:12],
        "user_id": user_id,
        "description": f"{user_id} agent",
    }


def test_default_cost_hashes_load_normally(tmp_path):
    config = tmp_path / "agents-auth.json"
    _write_v2_config(
        config,
        [
            _hashed_entry("ak_strong_key_value_111", "ceo", rounds=12),
            _hashed_entry("ak_strong_key_value_222", "research-agent", rounds=12),
        ],
    )

    mw = AgentAuthMiddleware(app=None, config_path=str(config))
    user_ids = {a["user_id"] for a in mw._agents}
    assert user_ids == {"ceo", "research-agent"}


def test_weak_hashes_are_rejected_not_loaded(tmp_path, caplog):
    """A rounds=4 hash for an otherwise-valid entry must NOT make it into
    the registry. The strong entry alongside it must continue to load.
    """
    config = tmp_path / "agents-auth.json"
    _write_v2_config(
        config,
        [
            _hashed_entry("ak_weak_key_value_AAA111", "weak-agent", rounds=4),
            _hashed_entry("ak_strong_key_value_BBB222", "strong-agent", rounds=12),
        ],
    )

    with caplog.at_level(logging.ERROR, logger="memos.api.middleware.agent_auth"):
        mw = AgentAuthMiddleware(app=None, config_path=str(config))

    user_ids = {a["user_id"] for a in mw._agents}
    assert "weak-agent" not in user_ids, "rounds=4 hash leaked into registry"
    assert "strong-agent" in user_ids, "strong hash got dropped along with weak one"

    # ERROR log mentions the rejected agent and the threshold so ops know
    # exactly what to rotate.
    rejection_logs = [r for r in caplog.records if "REJECTED" in r.getMessage()]
    assert rejection_logs, "no ERROR log for the rejected weak hash"
    msg = " ".join(r.getMessage() for r in rejection_logs)
    assert "weak-agent" in msg
    assert f"< {MIN_BCRYPT_COST}" in msg or f"<{MIN_BCRYPT_COST}" in msg or str(MIN_BCRYPT_COST) in msg


def test_at_min_cost_threshold_loads(tmp_path):
    """Cost == MIN_BCRYPT_COST must load (boundary condition; the guard
    is `<`, not `<=`). If we ever flip this, this test fails and forces
    a deliberate decision."""
    config = tmp_path / "agents-auth.json"
    _write_v2_config(
        config,
        [_hashed_entry("ak_threshold_key_111", "boundary-agent", rounds=MIN_BCRYPT_COST)],
    )

    mw = AgentAuthMiddleware(app=None, config_path=str(config))
    assert any(a["user_id"] == "boundary-agent" for a in mw._agents)


def test_unparseable_hash_is_rejected(tmp_path, caplog):
    config = tmp_path / "agents-auth.json"
    _write_v2_config(
        config,
        [
            {
                "key_hash": "not-a-real-bcrypt-hash",
                "key_prefix": "ak_corrupt_xx",
                "user_id": "corrupt-agent",
            },
            _hashed_entry("ak_good_key_value_CCC", "good-agent", rounds=12),
        ],
    )

    with caplog.at_level(logging.ERROR, logger="memos.api.middleware.agent_auth"):
        mw = AgentAuthMiddleware(app=None, config_path=str(config))

    user_ids = {a["user_id"] for a in mw._agents}
    assert "corrupt-agent" not in user_ids
    assert "good-agent" in user_ids

    msg = " ".join(r.getMessage() for r in caplog.records if "REJECTED" in r.getMessage())
    assert "corrupt-agent" in msg
    assert "unparseable" in msg.lower()


def test_admin_router_explicit_rounds_meets_min_cost():
    """Pin the admin router's hash factor to be >= the middleware's
    minimum. If a future refactor lowers BCRYPT_ROUNDS below
    MIN_BCRYPT_COST, every newly-created key would be unloadable on
    next restart. Failing this test catches that at PR time."""
    from memos.api.routers.admin_router import BCRYPT_ROUNDS

    assert BCRYPT_ROUNDS >= MIN_BCRYPT_COST, (
        f"admin_router.BCRYPT_ROUNDS ({BCRYPT_ROUNDS}) is below "
        f"agent_auth.MIN_BCRYPT_COST ({MIN_BCRYPT_COST}) — newly-created "
        f"keys would be rejected on next server restart"
    )
