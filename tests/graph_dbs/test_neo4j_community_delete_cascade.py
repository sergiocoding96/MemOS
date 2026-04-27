"""
Regression tests for Neo4jCommunityGraphDB delete -> vec_db cascade.

Bug 4 (2026-04-26 storage audit): delete_node_by_prams ran DETACH DELETE on
Neo4j but never called self.vec_db.delete(), leaving orphan points in Qdrant.
The sibling delete_node_by_mem_cube_id had the right pattern; we mirror it.

These tests construct the class via __new__ to bypass __init__ (which would
require a real Neo4j connection) and inject mocks for self.driver, self.vec_db,
self.config, and self.db_name.
"""

from unittest.mock import MagicMock

import pytest

from memos.graph_dbs.neo4j_community import Neo4jCommunityGraphDB


def _make_db():
    """Build a Neo4jCommunityGraphDB instance with mocked driver and vec_db."""
    db = Neo4jCommunityGraphDB.__new__(Neo4jCommunityGraphDB)
    db.driver = MagicMock()
    db.vec_db = MagicMock()
    db.db_name = "test_db"
    config = MagicMock()
    config.user_name = "default_user"
    config.use_multi_db = False
    db.config = config
    return db


def _session_mock(db):
    return db.driver.session.return_value.__enter__.return_value


# ──────────────────────────────────────────────────────────────────────────────
# Bug 4: delete_node_by_prams must cascade to vec_db
# ──────────────────────────────────────────────────────────────────────────────


class TestDeleteNodeByPramsVecDbCascade:
    def test_delete_by_memory_ids_cascades_to_vec_db(self):
        db = _make_db()
        session = _session_mock(db)

        # First session.run is the SELECT-ids; second is the DETACH DELETE.
        # Cypher results are iterable of records; records are dict-like.
        select_result = [{"id": "mem-1"}, {"id": "mem-2"}]
        delete_result = MagicMock()
        session.run.side_effect = [select_result, delete_result]

        deleted = db.delete_node_by_prams(memory_ids=["mem-1", "mem-2"])

        assert deleted == 2
        # vec_db.delete must have been called with exactly the resolved ids
        db.vec_db.delete.assert_called_once_with(["mem-1", "mem-2"])

    def test_delete_by_filter_cascades_resolved_ids(self):
        """Filter path: ids resolved via get_by_metadata, then deleted in both stores."""
        db = _make_db()
        # get_by_metadata is consulted before the session is opened
        db.get_by_metadata = MagicMock(return_value=["mem-9"])
        session = _session_mock(db)

        select_result = [{"id": "mem-9"}]
        session.run.side_effect = [select_result, MagicMock()]

        deleted = db.delete_node_by_prams(filter={"memory_type": "LongTermMemory"})

        assert deleted == 1
        db.vec_db.delete.assert_called_once_with(["mem-9"])

    def test_no_vec_db_call_when_nothing_matched(self):
        """Empty match → no vec_db.delete call (avoids deleting nothing in a loop)."""
        db = _make_db()
        session = _session_mock(db)
        session.run.side_effect = [[], MagicMock()]

        deleted = db.delete_node_by_prams(memory_ids=["mem-nope"])

        assert deleted == 0
        db.vec_db.delete.assert_not_called()

    def test_no_vec_db_call_when_no_inputs(self):
        """No memory_ids/file_ids/filter → early return, no DB work."""
        db = _make_db()
        deleted = db.delete_node_by_prams()
        assert deleted == 0
        db.driver.session.assert_not_called()
        db.vec_db.delete.assert_not_called()

    def test_vec_db_failure_does_not_break_neo4j_delete(self):
        """If Qdrant is briefly down, Neo4j delete still succeeds (logged warning)."""
        db = _make_db()
        session = _session_mock(db)
        session.run.side_effect = [[{"id": "mem-x"}], MagicMock()]
        db.vec_db.delete.side_effect = RuntimeError("qdrant unreachable")

        # Must not raise — we accept a vector orphan over a half-failed delete
        deleted = db.delete_node_by_prams(memory_ids=["mem-x"])

        assert deleted == 1
        db.vec_db.delete.assert_called_once_with(["mem-x"])


# ──────────────────────────────────────────────────────────────────────────────
# Pair test for the sibling that was already correct (regression guard).
# Per TASK.md: "if no test exists for the existing delete_node_by_mem_cube_id
# either, write the missing pair."
# ──────────────────────────────────────────────────────────────────────────────


class TestDeleteNodeByMemCubeIdVecDbCascade:
    def test_hard_delete_cascades_to_vec_db(self):
        db = _make_db()
        session = _session_mock(db)

        # First run: SELECT ids; second run: DETACH DELETE
        select_result = [{"id": "mem-a"}, {"id": "mem-b"}]
        delete_result = MagicMock()
        delete_result.consume.return_value.counters.nodes_deleted = 2
        session.run.side_effect = [select_result, delete_result]

        deleted = db.delete_node_by_mem_cube_id(
            mem_cube_id="cube-1", delete_record_id="rec-1", hard_delete=True
        )

        assert deleted == 2
        db.vec_db.delete.assert_called_once_with(["mem-a", "mem-b"])

    def test_soft_delete_does_not_touch_vec_db(self):
        """Soft delete only marks status; vectors stay (recoverable)."""
        db = _make_db()
        session = _session_mock(db)

        soft_result = MagicMock()
        soft_result.single.return_value = {"updated_count": 3}
        session.run.return_value = soft_result

        updated = db.delete_node_by_mem_cube_id(
            mem_cube_id="cube-1", delete_record_id="rec-1", hard_delete=False
        )

        assert updated == 3
        db.vec_db.delete.assert_not_called()

    def test_missing_required_params_returns_zero(self):
        db = _make_db()
        assert db.delete_node_by_mem_cube_id() == 0
        assert db.delete_node_by_mem_cube_id(mem_cube_id="x") == 0
        assert db.delete_node_by_mem_cube_id(delete_record_id="y") == 0
        db.driver.session.assert_not_called()
        db.vec_db.delete.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
