"""Unit tests for the dependency health probe registry."""

from __future__ import annotations

from memos.storage.dependency_health import DependencyHealth


class TestDependencyHealth:
    def test_unregistered_probe_returns_not_registered(self):
        h = DependencyHealth()
        s = h.probe("missing")
        assert s.ok is False
        assert "not registered" in (s.error or "")

    def test_required_failure_marks_overall_red(self):
        h = DependencyHealth()
        h.register("qdrant", lambda: (_ for _ in ()).throw(RuntimeError("conn refused")))
        statuses = h.probe_all()
        assert statuses["qdrant"].ok is False
        assert "RuntimeError" in (statuses["qdrant"].error or "")
        assert h.overall_ok() is False

    def test_optional_failure_does_not_mark_overall_red(self):
        h = DependencyHealth()
        h.register("ok", lambda: {"hello": "world"}, required=True)
        h.register(
            "noisy",
            lambda: (_ for _ in ()).throw(RuntimeError("flaky")),
            required=False,
        )
        assert h.overall_ok() is True
        payload = h.to_payload()
        assert payload["ok"] is True
        assert payload["deps"]["noisy"]["ok"] is False
        assert payload["deps"]["ok"]["detail"] == {"hello": "world"}

    def test_to_payload_shape(self):
        h = DependencyHealth()
        h.register("a", lambda: {})
        h.register("b", lambda: {})
        out = h.to_payload()
        assert set(out.keys()) == {"ok", "deps"}
        assert set(out["deps"].keys()) == {"a", "b"}
        for d in out["deps"].values():
            assert {"name", "ok", "required", "latency_ms"}.issubset(d.keys())
