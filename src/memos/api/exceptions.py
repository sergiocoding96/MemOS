import logging

from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from memos.storage.exceptions import (
    DependencyUnavailable,
    Neo4jUnavailable,
    QdrantUnavailable,
)


logger = logging.getLogger(__name__)


# Recognized neo4j connection-class exception names. Sniffed by class name so
# we don't import neo4j unconditionally (the API can be served from a
# build that uses a different graph backend).
_NEO4J_UNAVAILABLE_NAMES = frozenset(
    {
        "ServiceUnavailable",
        "RoutingServiceUnavailable",
        "WriteServiceUnavailable",
        "ReadServiceUnavailable",
        "IncompleteCommit",
        "DatabaseUnavailable",
        "ConnectionAcquisitionTimeoutError",
        "SessionExpired",
    }
)


def _classify_dependency_error(exc: BaseException) -> DependencyUnavailable | None:
    """If `exc` is a recognized dependency-down error, return a typed
    `DependencyUnavailable` for the API layer to surface as 503. Otherwise
    return None and let the regular handler chain run.
    """
    if isinstance(exc, DependencyUnavailable):
        return exc
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    # Neo4j driver exception → Neo4jUnavailable
    if module.startswith("neo4j") and name in _NEO4J_UNAVAILABLE_NAMES:
        return Neo4jUnavailable(f"Neo4j unreachable ({name}): {exc}", cause=exc)
    # Walk the cause chain
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _classify_dependency_error(cause)
    return None


class APIExceptionHandler:
    """Centralized exception handling for MemOS APIs."""

    @staticmethod
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        errors = exc.errors()
        path = request.url.path
        method = request.method

        readable_errors = []
        for err in errors:
            loc = " -> ".join(str(loc_i) for loc_i in err.get("loc", []))
            readable_errors.append(
                f"[{loc}] {err.get('msg', 'unknown error')} (type: {err.get('type', 'unknown')})"
            )

        logger.error(
            f"Validation error on {method} {path}: {readable_errors}, raw errors: {errors}"
        )
        return JSONResponse(
            status_code=422,
            content={
                "code": 422,
                "message": f"Parameter validation error on {method} {path}: {'; '.join(readable_errors)}",
                "detail": errors,
                "data": None,
            },
        )

    @staticmethod
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions globally."""
        logger.error(f"ValueError: {exc}")
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": str(exc), "data": None},
        )

    @staticmethod
    async def dependency_unavailable_handler(request: Request, exc: DependencyUnavailable):
        """Map storage-dependency outages to HTTP 503.

        Bug 2 fix: previously Qdrant/Neo4j outages bubbled up as generic
        500s, or worse, were swallowed by the async scheduler and returned
        200 with a silently-lost extraction. Now the caller sees an explicit
        503 naming which dependency is down, so the request can be retried
        cleanly.
        """
        logger.warning(
            f"Dependency unavailable on {request.method} {request.url.path}: "
            f"{exc.dep_name}: {exc}"
        )
        return JSONResponse(
            status_code=503,
            content={
                "code": 503,
                "message": str(exc),
                "dependency": exc.dep_name,
                "data": None,
            },
            headers={"Retry-After": "5"},
        )

    @staticmethod
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions globally.

        First attempts to classify `exc` as a storage-dependency outage; if
        so, fall through to the 503 handler. Otherwise return 500.
        """
        dep_exc = _classify_dependency_error(exc)
        if dep_exc is not None:
            return await APIExceptionHandler.dependency_unavailable_handler(request, dep_exc)
        logger.error(f"Exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": str(exc), "data": None},
        )

    @staticmethod
    async def http_error_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions globally."""
        logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": str(exc.detail), "data": None},
        )
