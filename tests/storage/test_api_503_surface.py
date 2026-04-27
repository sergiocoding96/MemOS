"""Unit tests for the API's 503-on-dependency-down behavior.

Builds a tiny FastAPI app that re-uses the real APIExceptionHandler +
DependencyUnavailable handlers. Avoids importing memos.api.server_api
(which pulls in all middleware + routers).
"""

from __future__ import annotations

import pytest

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from memos.api.exceptions import APIExceptionHandler
from memos.storage.exceptions import (
    DependencyUnavailable,
    Neo4jUnavailable,
    QdrantUnavailable,
)


# Synthetic neo4j ServiceUnavailable so we don't need the real driver
class _FakeNeo4jServiceUnavailable(Exception):
    pass


_FakeNeo4jServiceUnavailable.__module__ = "neo4j.exceptions"
_FakeNeo4jServiceUnavailable.__name__ = "ServiceUnavailable"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/qdrant-down")
    def qdrant_down():
        raise QdrantUnavailable("simulated outage")

    @app.get("/neo4j-down")
    def neo4j_down():
        raise Neo4jUnavailable("simulated outage")

    @app.get("/raw-neo4j")
    def raw_neo4j():
        raise _FakeNeo4jServiceUnavailable("conn refused")

    @app.get("/value-error")
    def value_error():
        raise ValueError("bad input")

    @app.get("/random-500")
    def random_500():
        raise RuntimeError("kaboom")

    app.exception_handler(RequestValidationError)(APIExceptionHandler.validation_error_handler)
    app.exception_handler(ValueError)(APIExceptionHandler.value_error_handler)
    app.exception_handler(HTTPException)(APIExceptionHandler.http_error_handler)
    app.exception_handler(DependencyUnavailable)(
        APIExceptionHandler.dependency_unavailable_handler
    )
    app.exception_handler(Exception)(APIExceptionHandler.global_exception_handler)
    return TestClient(app, raise_server_exceptions=False)


class TestApi503:
    def test_qdrant_unavailable_returns_503_with_dep_name(self, client: TestClient):
        r = client.get("/qdrant-down")
        assert r.status_code == 503
        body = r.json()
        assert body["code"] == 503
        assert body["dependency"] == "qdrant"
        assert "simulated" in body["message"]
        # Retry-After header is set so polite clients can back off
        assert r.headers.get("Retry-After") == "5"

    def test_neo4j_unavailable_returns_503(self, client: TestClient):
        r = client.get("/neo4j-down")
        assert r.status_code == 503
        body = r.json()
        assert body["dependency"] == "neo4j"

    def test_raw_neo4j_service_unavailable_classified_to_503(self, client: TestClient):
        # The driver raised a vanilla neo4j.exceptions.ServiceUnavailable —
        # the global handler must classify it instead of bubbling 500.
        r = client.get("/raw-neo4j")
        assert r.status_code == 503
        body = r.json()
        assert body["dependency"] == "neo4j"
        assert "conn refused" in body["message"]

    def test_value_error_still_400(self, client: TestClient):
        r = client.get("/value-error")
        assert r.status_code == 400

    def test_unrelated_exception_still_500(self, client: TestClient):
        r = client.get("/random-500")
        assert r.status_code == 500
        assert r.json()["code"] == 500
