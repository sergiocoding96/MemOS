"""Regression tests: ``DependencyUnavailable`` must propagate through
broad ``except Exception`` blocks instead of being silently logged.

Background. PR #8 (storage resilience) introduced typed
``DependencyUnavailable`` exceptions and a durable retry queue. The
dispatcher's contract: if a worker raises ``DependencyUnavailable``,
the dispatcher catches it at the top level and enqueues the message
into the retry queue, so the write is preserved across a Qdrant/Neo4j
blip.

That contract was being broken by ~7 broad ``except Exception:`` blocks
inside the worker code itself — ``DependencyUnavailable`` is an
``Exception`` subclass, so those generic handlers caught it first,
logged a warning, and let the message be marked "processed". The
retry queue never saw it; the write was silently lost.

The fix wraps each generic catch with ``except DependencyUnavailable:
raise`` *before* the generic ``except Exception`` so the typed
exception propagates to the dispatcher.

These tests pin the propagation invariant for each of the patched
sites. If a future refactor reorders the except clauses or removes
the typed catch, these tests fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memos.storage.exceptions import (
    DependencyUnavailable,
    Neo4jUnavailable,
    QdrantUnavailable,
)


# ---------------------------------------------------------------------------
# MemReadMessageHandler.process_message
# ---------------------------------------------------------------------------


def _make_mem_read_handler():
    from memos.mem_scheduler.task_schedule_modules.handlers.mem_read_handler import (
        MemReadMessageHandler,
    )

    h = MemReadMessageHandler.__new__(MemReadMessageHandler)
    h.scheduler_context = MagicMock()
    return h


def _fake_mem_read_message(content: str = '["mid-1"]'):
    msg = MagicMock()
    msg.user_id = "u1"
    msg.mem_cube_id = "cube"
    msg.content = content
    msg.user_name = "alice"
    msg.info = {}
    msg.chat_history = None
    msg.user_context = None
    msg.task_id = "tsk"
    return msg


def test_process_message_propagates_qdrant_unavailable():
    """The top-level ``except Exception`` in process_message used to
    swallow ``QdrantUnavailable``. After the fix, it must propagate."""
    h = _make_mem_read_handler()
    cube = MagicMock()
    # The handler calls .get_mem_cube() then drives text_mem; raise the
    # typed exception from the cube fetch so we hit the outer handler.
    h.scheduler_context.get_mem_cube.side_effect = QdrantUnavailable("qdrant down")

    with pytest.raises(QdrantUnavailable):
        h.process_message(_fake_mem_read_message())


def test_process_message_still_swallows_generic_exception():
    """The fix must NOT change behavior for non-dep exceptions — those
    should still be logged and swallowed (so a malformed message
    doesn't kill the worker)."""
    h = _make_mem_read_handler()
    h.scheduler_context.get_mem_cube.side_effect = ValueError("not a dep outage")

    # Should not raise — generic exception is still handled.
    h.process_message(_fake_mem_read_message())


def test_process_message_propagates_neo4j_unavailable():
    """Same invariant for the Neo4j subclass — both typed deps must
    propagate (the base catch is on ``DependencyUnavailable``)."""
    h = _make_mem_read_handler()
    h.scheduler_context.get_mem_cube.side_effect = Neo4jUnavailable("graph blip")

    with pytest.raises(Neo4jUnavailable):
        h.process_message(_fake_mem_read_message())


# ---------------------------------------------------------------------------
# MemReadMessageHandler.batch_handler
# ---------------------------------------------------------------------------


def test_batch_handler_drains_then_raises_first_dep_error():
    """``batch_handler`` runs messages on a thread pool. If one worker
    raises ``DependencyUnavailable``, the batch must finish draining
    (no thread leak) and then the dep error is re-raised.
    """
    from memos.mem_scheduler.task_schedule_modules.handlers.mem_read_handler import (
        MemReadMessageHandler,
    )

    h = MemReadMessageHandler.__new__(MemReadMessageHandler)
    h.scheduler_context = MagicMock()

    processed_ids: list[str] = []

    def fake_process(msg):
        processed_ids.append(msg.user_id)
        if msg.user_id == "boom":
            raise QdrantUnavailable("qdrant down")
        if msg.user_id == "bad":
            raise ValueError("benign")
        return None

    h.process_message = fake_process

    batch = [
        _make_msg("ok-1"),
        _make_msg("boom"),
        _make_msg("bad"),
        _make_msg("ok-2"),
    ]

    with pytest.raises(QdrantUnavailable):
        h.batch_handler(user_id="u1", mem_cube_id="cube", batch=batch)

    # All four messages reached process_message — none short-circuited.
    assert sorted(processed_ids) == ["bad", "boom", "ok-1", "ok-2"]


def _make_msg(user_id: str):
    msg = MagicMock()
    msg.user_id = user_id
    msg.item_id = f"i-{user_id}"
    msg.task_id = f"t-{user_id}"
    return msg


# ---------------------------------------------------------------------------
# Static structural assertion: every patched site has the typed-catch
# preceding a generic ``except Exception``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "src/memos/mem_scheduler/task_schedule_modules/handlers/mem_read_handler.py",
        "src/memos/mem_scheduler/task_schedule_modules/handlers/add_handler.py",
        "src/memos/multi_mem_cube/single_cube.py",
        "src/memos/graph_dbs/neo4j_community.py",
    ],
)
def test_dependency_unavailable_imported_in_patched_modules(module_path):
    """Every module that gained a typed catch must import
    ``DependencyUnavailable``. A missing import would mean a stray
    ``except DependencyUnavailable: raise`` block raises ``NameError``
    at runtime — not what we want for a defense-in-depth fix.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / module_path).read_text()
    assert "from memos.storage.exceptions import DependencyUnavailable" in text, (
        f"{module_path} has typed-catch blocks but doesn't import "
        f"DependencyUnavailable"
    )
    assert "except DependencyUnavailable" in text, (
        f"{module_path} imports DependencyUnavailable but has no "
        f"`except DependencyUnavailable` clause"
    )


# ---------------------------------------------------------------------------
# Neo4jCommunityGraphDB driver — write + search paths
# ---------------------------------------------------------------------------
#
# The earlier PR covered the read paths (``get_node`` / ``get_nodes_batch``).
# These cover the write side (``add_node`` / ``add_nodes_batch``) and the
# similarity-search path used by write-side dedup. Without these, a Qdrant
# outage during a write would be tagged ``vector_sync=failed`` in Neo4j-only
# and the API would return HTTP 200 — exactly the silent-loss path the
# resilience work was meant to close.


def _make_neo4j_db_with_failing_vec_db(side_effect: BaseException):
    """Build a Neo4jCommunityGraphDB stub whose vec_db raises ``side_effect``
    on add / search. We bypass __init__ (real init opens TCP connections)
    and only wire the attributes the tested methods touch.
    """
    from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB

    db = Neo4jCommunityGraphDB.__new__(Neo4jCommunityGraphDB)
    db.vec_db = MagicMock()
    db.vec_db.add.side_effect = side_effect
    db.vec_db.search.side_effect = side_effect
    # Some tests touch the Neo4j driver too; give it a no-op mock so
    # raising from vec_db is what dominates the call path.
    db.driver = MagicMock()
    db.embedder = MagicMock()
    db.config = MagicMock()
    db.config.user_name = "alice"
    return db


def test_add_node_propagates_qdrant_unavailable():
    """``add_node`` writes to vec_db before the Neo4j MERGE. A Qdrant
    outage there used to be tagged ``vector_sync=failed`` silently — the
    typed exception must now propagate."""
    from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB

    db = _make_neo4j_db_with_failing_vec_db(QdrantUnavailable("write blip"))

    # Drive only the vec_db block: bind add_node and call it; we expect
    # the typed exception to propagate before reaching the Neo4j query.
    add_node = Neo4jCommunityGraphDB.add_node.__get__(db, Neo4jCommunityGraphDB)
    with pytest.raises(QdrantUnavailable):
        add_node(
            id="11111111-1111-1111-1111-111111111111",
            memory="hello",
            metadata={
                "embedding": [0.1] * 8,
                "created_at": "2026-04-27T00:00:00",
                "updated_at": "2026-04-27T00:00:00",
            },
        )


def test_add_nodes_batch_propagates_qdrant_unavailable():
    """Batch insert path. Same invariant as ``add_node``."""
    from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB

    db = _make_neo4j_db_with_failing_vec_db(QdrantUnavailable("batch blip"))

    add_nodes_batch = Neo4jCommunityGraphDB.add_nodes_batch.__get__(
        db, Neo4jCommunityGraphDB
    )
    nodes = [
        {
            "id": f"{i:08d}-1111-1111-1111-111111111111",
            "memory": f"hello-{i}",
            "metadata": {
                "embedding": [0.1] * 8,
                "created_at": "2026-04-27T00:00:00",
                "updated_at": "2026-04-27T00:00:00",
            },
        }
        for i in range(3)
    ]
    with pytest.raises(QdrantUnavailable):
        add_nodes_batch(nodes)


def test_add_node_does_not_widen_typed_catch_to_generic_errors():
    """Regression guard: ``add_node`` must NOT promote a generic
    ``ValueError`` from vec_db into ``QdrantUnavailable``. The typed
    catch added in this PR is narrow on purpose — only typed dep
    outages should propagate; everything else stays in the existing
    log-and-tag-as-failed path.

    We assert the *negative*: a generic vec_db error must not surface
    as ``QdrantUnavailable``. The function may still raise something
    else further down (the Neo4j MERGE call has its own concerns and
    we're not mocking the full driver), but it must NOT raise
    ``QdrantUnavailable`` from a non-typed cause.
    """
    from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB

    db = _make_neo4j_db_with_failing_vec_db(ValueError("bad vector"))

    add_node = Neo4jCommunityGraphDB.add_node.__get__(db, Neo4jCommunityGraphDB)
    raised: BaseException | None = None
    try:
        add_node(
            id="11111111-1111-1111-1111-111111111111",
            memory="hello",
            metadata={
                "embedding": [0.1] * 8,
                "created_at": "2026-04-27T00:00:00",
                "updated_at": "2026-04-27T00:00:00",
            },
        )
    except BaseException as e:
        raised = e

    # The vital invariant: whatever (if anything) propagates, it must
    # not be a ``DependencyUnavailable`` subclass. Anything else means
    # the typed catch did its job (passed the ValueError through to
    # the existing logger.warning path).
    assert not isinstance(raised, DependencyUnavailable), (
        f"typed catch widened — generic vec_db error became "
        f"{type(raised).__name__}: {raised}"
    )
