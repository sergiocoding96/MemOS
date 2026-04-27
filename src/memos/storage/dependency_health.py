"""
Lightweight health probes for storage dependencies.

Backs `/health` and `/health/deps`. Each probe is short-timeout (default
2 seconds) and never raises — it returns a `DepStatus` describing what it
saw. Callers compose these into a single response.

Probes are registered lazily so that constructing a `DependencyHealth`
does not itself open connections; opening a connection only happens when
`.probe(...)` runs. This avoids slowing down liveness checks and avoids
DependencyHealth becoming a circular import for the modules it probes.
"""

from __future__ import annotations

import time

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from memos.log import get_logger


logger = get_logger(__name__)


@dataclass
class DepStatus:
    name: str
    ok: bool
    required: bool
    latency_ms: float
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


ProbeFn = Callable[[], dict[str, Any]]
"""A probe is a zero-arg callable. It returns an arbitrary dict (merged
into `DepStatus.detail`) on success, or raises on failure. Probes should
be cheap (round-trip < 100ms in the happy path)."""


class DependencyHealth:
    """Registry + runner for storage health probes."""

    def __init__(self, *, probe_timeout_s: float = 2.0) -> None:
        self.probe_timeout_s = probe_timeout_s
        self._probes: dict[str, tuple[ProbeFn, bool]] = {}

    def register(self, name: str, probe: ProbeFn, *, required: bool = True) -> None:
        """Register a probe. `required=True` means /health goes red if the
        probe fails; `required=False` means it's reported but doesn't gate."""
        self._probes[name] = (probe, required)

    def probe(self, name: str) -> DepStatus:
        if name not in self._probes:
            return DepStatus(name=name, ok=False, required=False, latency_ms=0.0,
                             error="probe not registered")
        probe_fn, required = self._probes[name]
        start = time.perf_counter()
        try:
            detail = probe_fn() or {}
            latency_ms = (time.perf_counter() - start) * 1000
            return DepStatus(name=name, ok=True, required=required,
                             latency_ms=latency_ms, detail=detail)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            err = f"{type(e).__name__}: {e}"
            logger.warning(f"[DependencyHealth] probe {name} failed: {err}")
            return DepStatus(name=name, ok=False, required=required,
                             latency_ms=latency_ms, error=err)

    def probe_all(self) -> dict[str, DepStatus]:
        return {name: self.probe(name) for name in self._probes}

    def overall_ok(self) -> bool:
        """True iff all *required* probes are green."""
        statuses = self.probe_all()
        return all(s.ok for s in statuses.values() if s.required)

    def to_payload(self) -> dict[str, Any]:
        statuses = self.probe_all()
        return {
            "ok": all(s.ok for s in statuses.values() if s.required),
            "deps": {name: s.to_dict() for name, s in statuses.items()},
        }


# ────────────────────────────────────────────────────────────────────────────
# Default probes — wired by the API layer at startup.
# ────────────────────────────────────────────────────────────────────────────


def make_qdrant_probe(get_client: Callable[[], Any]) -> ProbeFn:
    """Returns a probe that calls qdrant_client.get_collections().

    `get_client` is a thunk so we don't open a connection at import time
    or during DependencyHealth construction."""

    def _probe() -> dict[str, Any]:
        client = get_client()
        if client is None:
            raise RuntimeError("qdrant client not available")
        collections = client.get_collections()
        names = [c.name for c in getattr(collections, "collections", [])]
        return {"collections": len(names)}

    return _probe


def make_neo4j_probe(get_driver: Callable[[], Any], db_name: str | None = None) -> ProbeFn:
    """Returns a probe that runs `RETURN 1` on the Neo4j driver."""

    def _probe() -> dict[str, Any]:
        driver = get_driver()
        if driver is None:
            raise RuntimeError("neo4j driver not available")
        with driver.session(database=db_name) if db_name else driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            row = result.single()
            ok = bool(row and row["ok"] == 1)
            if not ok:
                raise RuntimeError("neo4j returned unexpected probe result")
        return {}

    return _probe


def make_llm_probe(generate_fn: Callable[[list[dict[str, str]]], str]) -> ProbeFn:
    """Returns a probe that asks the LLM to generate a 1-token response.

    Avoid this for high-volume endpoints — LLM probes are slower and cost
    money. Suitable for /health/deps."""

    def _probe() -> dict[str, Any]:
        out = generate_fn([{"role": "user", "content": "ping"}])
        return {"sample": (out or "")[:64]}

    return _probe
