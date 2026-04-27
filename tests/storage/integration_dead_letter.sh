#!/usr/bin/env bash
# Bug 2 integration test: async write that fails repeatedly lands in
# the SQLite dead-letter table after ~10 attempts. Operator-run.
set -euo pipefail

API="${MEMOS_API_URL:-http://localhost:8001}"
KEY="${MEMOS_AGENT_KEY:?MEMOS_AGENT_KEY must be set to a valid raw bearer key}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"
QUEUE_DB="${MEMOS_RETRY_QUEUE_PATH:-$HOME/.hermes/state/retry_queue.sqlite}"
USER_ID="${INTEG_USER:-integ-deadletter-user}"
CUBE="${INTEG_CUBE:-integ-deadletter-cube}"

step() { echo -e "\n=== $* ==="; }
fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

[[ -f "$QUEUE_DB" ]] || fail "retry queue DB not found at $QUEUE_DB; ensure server is running with retry queue wired"

step "1) Snapshot dead-letter row count before"
DL_BEFORE=$(sqlite3 "$QUEUE_DB" "SELECT COUNT(*) FROM dead_letter")
echo "dead_letter rows before: $DL_BEFORE"

step "2) Submit one async write (will become a retry candidate)"
RESP=$(curl -sS -X POST "$API/product/add" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"user_id\":\"$USER_ID\",\"writable_cube_ids\":[\"$CUBE\"],\"messages\":[{\"role\":\"user\",\"content\":\"dead-letter test\"}],\"async_mode\":\"async\",\"mode\":\"fast\"}")
echo "submit response: $(echo "$RESP" | head -c 400)"

step "3) Stop Qdrant for the retry window (sleep 600s)"
docker stop "$QDRANT_CONTAINER" >/dev/null
echo "Sleeping 600s for the retry queue to exhaust max_attempts..."
sleep 600

step "4) Inspect dead-letter table"
DL_AFTER=$(sqlite3 "$QUEUE_DB" "SELECT COUNT(*) FROM dead_letter")
echo "dead_letter rows after: $DL_AFTER"
DELTA=$((DL_AFTER - DL_BEFORE))
echo "delta=$DELTA"
[[ "$DELTA" -ge 1 ]] || fail "expected at least 1 new dead-letter row, got $DELTA"
sqlite3 -header -column "$QUEUE_DB" \
  "SELECT id, label, attempts, substr(last_error,1,80) AS last_err FROM dead_letter ORDER BY dead_at DESC LIMIT 3"
pass "dead-letter row(s) recorded with original error"

step "5) Bring Qdrant back"
docker start "$QDRANT_CONTAINER" >/dev/null
sleep 5

echo -e "\n✅ integration_dead_letter.sh PASSED"
