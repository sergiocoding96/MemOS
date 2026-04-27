import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from memos.api.exceptions import APIExceptionHandler
from memos.api.middleware.agent_auth import AgentAuthMiddleware
from memos.api.middleware.rate_limit import RateLimitMiddleware
from memos.api.middleware.request_context import RequestContextMiddleware
from memos.api.routers.admin_router import router as admin_router
from memos.api.routers.server_router import router as server_router
from memos.storage.dependency_health import DependencyHealth
from memos.storage.exceptions import DependencyUnavailable


load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MemOS Server REST APIs",
    description="A REST API for managing multiple users with MemOS Server.",
    version="1.0.1",
)

app.mount("/download", StaticFiles(directory=os.getenv("FILE_LOCAL_PATH")), name="static_mapping")

# Middleware execution order (outermost first):
# 1. RateLimitMiddleware — reject excessive requests before any processing
# 2. AgentAuthMiddleware — validate per-agent API keys, bind user_id to context
# 3. RequestContextMiddleware — inject trace_id, log request metadata
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AgentAuthMiddleware)
app.add_middleware(RequestContextMiddleware, source="server_api")
# Include routers
app.include_router(server_router)
app.include_router(admin_router)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoints
#
# Pre-Bug-2, /health was a static "healthy" string that returned 200 even when
# Qdrant 401'd or Neo4j was down. Container schedulers therefore happily kept
# routing traffic at a broken instance. /health now actually probes the
# dependencies, and /health/deps gives operators a per-dep breakdown.
#
# Probes are registered lazily so that constructing the DependencyHealth
# registry doesn't trigger connections at import time. The default registry
# is empty; the first /health call wires probes via _ensure_health_probes()
# using the same singletons the rest of the API uses.
# ─────────────────────────────────────────────────────────────────────────────

dependency_health = DependencyHealth(probe_timeout_s=2.0)
_health_probes_wired = False


def _ensure_health_probes() -> None:
    """Lazy probe registration. Idempotent."""
    global _health_probes_wired
    if _health_probes_wired:
        return
    try:
        from memos.storage.dependency_health import (
            make_neo4j_probe,
            make_qdrant_probe,
        )

        # Qdrant probe: short-lived client per probe call so we don't keep a
        # long-lived connection of our own. Env vars match what the rest of
        # the api/config module reads (QDRANT_HOST/QDRANT_PORT/QDRANT_API_KEY
        # or QDRANT_URL).
        def _qdrant_client():
            try:
                from qdrant_client import QdrantClient

                kwargs = {"timeout": 2.0}
                qdrant_url = os.getenv("QDRANT_URL")
                if qdrant_url:
                    kwargs["url"] = qdrant_url
                else:
                    kwargs.update(
                        host=os.getenv("QDRANT_HOST", "localhost"),
                        port=int(os.getenv("QDRANT_PORT", "6333")),
                        https=False,
                    )
                api_key = os.getenv("QDRANT_API_KEY")
                if api_key:
                    kwargs["api_key"] = api_key
                return QdrantClient(**kwargs)
            except Exception as e:  # pragma: no cover — best effort
                logger.warning(f"qdrant probe client construction failed: {e}")
                return None

        dependency_health.register("qdrant", make_qdrant_probe(_qdrant_client), required=True)

        # Neo4j probe
        def _neo4j_driver():
            try:
                from neo4j import GraphDatabase

                uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "")
                return GraphDatabase.driver(
                    uri,
                    auth=(user, password),
                    connection_timeout=2.0,
                    max_transaction_retry_time=2.0,
                )
            except Exception as e:  # pragma: no cover — best effort
                logger.warning(f"neo4j probe driver construction failed: {e}")
                return None

        dependency_health.register(
            "neo4j",
            make_neo4j_probe(_neo4j_driver, db_name=os.getenv("NEO4J_DB_NAME")),
            required=True,
        )
    except Exception as e:
        # Don't crash the app because probe registration failed; /health/deps
        # will simply report "probe not registered" for the missing pieces.
        logger.warning(f"_ensure_health_probes failed: {e}")
    _health_probes_wired = True


@app.get("/health")
def health_check():
    """Container and load balancer health endpoint.

    Returns 200 only when every required storage dependency is reachable.
    Returns 503 (with the failing dep names) otherwise. This replaces the
    prior implementation which always returned 200 even when Qdrant 401'd
    or Neo4j was down.
    """
    _ensure_health_probes()
    payload = dependency_health.to_payload()
    base = {
        "status": "healthy" if payload["ok"] else "degraded",
        "service": "memos",
        "version": app.version,
    }
    if payload["ok"]:
        return base
    failing = [name for name, dep in payload["deps"].items()
               if dep["required"] and not dep["ok"]]
    return JSONResponse(
        status_code=503,
        content={
            **base,
            "failing_dependencies": failing,
        },
        headers={"Retry-After": "5"},
    )


@app.get("/health/deps")
def health_deps():
    """Per-dependency status and latency. Always returns 200; the body
    indicates which deps are red. Operators (and dashboards) read this to
    triage outages."""
    _ensure_health_probes()
    return dependency_health.to_payload()


# Request validation failed
app.exception_handler(RequestValidationError)(APIExceptionHandler.validation_error_handler)
# Invalid business code parameters
app.exception_handler(ValueError)(APIExceptionHandler.value_error_handler)
# Business layer manual exception
app.exception_handler(HTTPException)(APIExceptionHandler.http_error_handler)
# Storage dependency outage → 503 with named dependency
app.exception_handler(DependencyUnavailable)(APIExceptionHandler.dependency_unavailable_handler)
# Fallback for unknown errors (also classifies neo4j connection errors → 503)
app.exception_handler(Exception)(APIExceptionHandler.global_exception_handler)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    bind_host = os.getenv("MEMOS_BIND_HOST", "127.0.0.1")
    uvicorn.run("memos.api.server_api:app", host=bind_host, port=args.port, workers=args.workers)
