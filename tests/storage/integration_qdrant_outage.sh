#!/usr/bin/env bash
# Bug 2 integration test: sync write returns 503 when Qdrant is down,
# then recovers when Qdrant comes back. Operator-run; not pytest collected.
set -euo pipefail

API="${MEMOS_API_URL:-http://localhost:8001}"
KEY="${MEMOS_AGENT_KEY:?MEMOS_AGENT_KEY must be set to a valid raw bearer key}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"
USER_ID="${INTEG_USER:-integ-storage-user}"
CUBE="${INTEG_CUBE:-integ-storage-cube}"

step() { echo -e "\n=== $* ==="; }
fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

step "1) Stop Qdrant"
docker stop "$QDRANT_CONTAINER" >/dev/null
sleep 2

step "2) POST /product/add — expect 503"
HTTP=$(curl -s -o /tmp/add.body -w '%{http_code}' \
  -X POST "$API/product/add" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"$USER_ID\",\"writable_cube_ids\":[\"$CUBE\"],\"messages\":[{\"role\":\"user\",\"content\":\"503 test\"}],\"async_mode\":\"sync\",\"mode\":\"fast\"}")
echo "HTTP=$HTTP body=$(cat /tmp/add.body)"
[[ "$HTTP" == "503" ]] || fail "expected 503 with Qdrant down, got $HTTP"
DEP=$(jq -r '.dependency' /tmp/add.body)
[[ "$DEP" == "qdrant" ]] || fail "expected dependency=qdrant in body, got $DEP"
pass "503 surfaced cleanly"

step "3) Start Qdrant"
docker start "$QDRANT_CONTAINER" >/dev/null
echo "Waiting for /health/deps to report qdrant green..."
for _ in $(seq 1 30); do
  STATUS=$(curl -s "$API/health/deps" | jq -r '.deps.qdrant.ok' 2>/dev/null || echo "false")
  [[ "$STATUS" == "true" ]] && break
  sleep 1
done
[[ "$STATUS" == "true" ]] || fail "qdrant did not come back within 30s"

step "4) POST /product/add — expect 200"
HTTP=$(curl -s -o /tmp/add.body -w '%{http_code}' \
  -X POST "$API/product/add" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"$USER_ID\",\"writable_cube_ids\":[\"$CUBE\"],\"messages\":[{\"role\":\"user\",\"content\":\"recovery test\"}],\"async_mode\":\"sync\",\"mode\":\"fast\"}")
echo "HTTP=$HTTP body=$(cat /tmp/add.body | head -c 400)"
[[ "$HTTP" == "200" ]] || fail "expected 200 after Qdrant recovered, got $HTTP"
pass "Recovery write returned 200"

echo -e "\n✅ integration_qdrant_outage.sh PASSED"
