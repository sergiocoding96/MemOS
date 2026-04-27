# Bug 2 integration tests — operator-run

These scripts exercise the storage-resilience fixes against live Qdrant +
Neo4j containers. They are NOT pytest collected by default because they
require docker access and take 10-12 minutes wall-clock total.

## Prerequisites

- `docker compose` with the project's standard Qdrant + Neo4j stack.
- MemOS API server running on `localhost:8001` with the demo agents wired
  (research-agent or any other auth'd agent works).
- `MEMOS_AGENT_KEY` env var set to a valid Bearer key for the demo agent.
- `jq`, `curl`, `docker` on PATH.

## Run

```bash
export MEMOS_AGENT_KEY=...   # raw key for an authenticated agent
bash tests/storage/integration_qdrant_outage.sh
bash tests/storage/integration_neo4j_outage.sh
bash tests/storage/integration_dead_letter.sh
bash tests/storage/integration_health_deps.sh
```

Each script prints a `PASS` / `FAIL` line at the end. A non-zero exit code
indicates failure.

## What each test asserts

### `integration_qdrant_outage.sh` — sync write 503 + recovery
1. `docker stop qdrant` — confirm Qdrant is unreachable.
2. POST `/product/add` — assert HTTP **503** with body `{"dependency":"qdrant"}`.
3. `docker start qdrant` — wait until /health/deps reports green.
4. POST `/product/add` again — assert HTTP **200** and the memory exists.

### `integration_neo4j_outage.sh` — same as above, for Neo4j.

### `integration_dead_letter.sh` — durable retry → dead-letter
1. Submit one async write (mode=async).
2. `docker stop qdrant` and keep it down for 600s.
3. Wait for the SQLite retry queue worker to exhaust ~10 attempts
   (1+2+4+8+16+32+60×4 ≈ 363s, well within 600s).
4. Inspect the SQLite dead-letter table — assert exactly one row exists
   for the submitted memory id.
5. `docker start qdrant` — confirm subsequent writes succeed normally.

### `integration_health_deps.sh` — `/health` accuracy
1. `docker stop qdrant` — assert `/health` returns **503** with
   `failing_dependencies: ["qdrant"]`. Pre-fix it returned 200.
2. `docker start qdrant`, wait — assert `/health` returns 200 again.
3. Hit `/health/deps` — assert per-dep latency_ms is reported.
