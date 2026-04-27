#!/usr/bin/env bash
# Bug 2 integration test: /health is now real (returns 503 when a required
# dep is down) and /health/deps reports per-dep status. Operator-run.
set -euo pipefail

API="${MEMOS_API_URL:-http://localhost:8001}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"

step() { echo -e "\n=== $* ==="; }
fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

step "1) Baseline: /health should be 200 with everything up"
HTTP=$(curl -s -o /tmp/h.body -w '%{http_code}' "$API/health")
echo "HTTP=$HTTP body=$(cat /tmp/h.body)"
[[ "$HTTP" == "200" ]] || fail "expected /health 200 baseline, got $HTTP"
pass "/health 200 baseline"

step "2) Stop Qdrant"
docker stop "$QDRANT_CONTAINER" >/dev/null
sleep 2

step "3) /health should now be 503 with failing_dependencies=[qdrant]"
HTTP=$(curl -s -o /tmp/h.body -w '%{http_code}' "$API/health")
echo "HTTP=$HTTP body=$(cat /tmp/h.body)"
[[ "$HTTP" == "503" ]] || fail "expected /health 503 while Qdrant down, got $HTTP"
FAILING=$(jq -r '.failing_dependencies[]?' /tmp/h.body)
echo "$FAILING" | grep -q '^qdrant$' || fail "expected 'qdrant' in failing_dependencies"
pass "/health 503 + qdrant flagged"

step "4) /health/deps detail breakdown"
curl -s "$API/health/deps" | jq .
QDRANT_OK=$(curl -s "$API/health/deps" | jq -r '.deps.qdrant.ok')
[[ "$QDRANT_OK" == "false" ]] || fail "/health/deps.deps.qdrant.ok must be false"
pass "/health/deps reports qdrant red with latency"

step "5) Restart Qdrant + recover"
docker start "$QDRANT_CONTAINER" >/dev/null
for _ in $(seq 1 30); do
  HTTP=$(curl -s -o /dev/null -w '%{http_code}' "$API/health")
  [[ "$HTTP" == "200" ]] && break
  sleep 1
done
[[ "$HTTP" == "200" ]] || fail "expected /health to recover to 200, got $HTTP"
pass "/health recovered to 200"

echo -e "\n✅ integration_health_deps.sh PASSED"
