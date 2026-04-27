#!/usr/bin/env bash
# start-memos.sh — Decrypt secrets, load env, start MemOS server
# Usage: ./start-memos.sh [--port 8001]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGE_KEY="${MEMOS_AGE_KEY:-$HOME/.memos/keys/memos.key}"
SECRETS_ENC="${MEMOS_SECRETS:-$HOME/.memos/secrets.env.age}"
AGE_BIN="${AGE_BIN:-$(command -v age || echo /home/linuxbrew/.linuxbrew/bin/age)}"
PORT="${1:-8001}"

# 0. Tighten file-system permissions on every restart (defense-in-depth
#    for F-07 / F-08 from the 2026-04-26 zero-knowledge audit). Idempotent
#    by design — `chmod` always writes the requested mode, so this self-
#    heals if perms drift (e.g. someone runs the server under a different
#    umask, or copies a backup back in with default modes).
_harden_perms() {
    # Owner-only on the data dir tree: SQLite DBs hold bcrypt key hashes
    # (memos_users.db / agents-auth.json) and rate-limit + retry-queue
    # state. Don't expose to other local users.
    if [[ -d "$HOME/.memos" ]]; then
        chmod 700 "$HOME/.memos" 2>/dev/null || true
        # Tighten every regular file under ~/.memos/ to 600.
        find "$HOME/.memos" -type f -exec chmod 600 {} + 2>/dev/null || true
        # Subdirs to 700 (listable only by owner).
        find "$HOME/.memos" -mindepth 1 -type d -exec chmod 700 {} + 2>/dev/null || true
    fi
    # Logs: the file currently in use may be held open by the previous
    # server process — `chmod` still works on an open file. Logs hold
    # request paths, stack traces, and trace ids that are sensitive even
    # after redaction. Owner-only.
    if [[ -d "$SCRIPT_DIR/.memos" ]]; then
        chmod 700 "$SCRIPT_DIR/.memos" 2>/dev/null || true
        find "$SCRIPT_DIR/.memos" -type d -exec chmod 700 {} + 2>/dev/null || true
        find "$SCRIPT_DIR/.memos" -type f -exec chmod 600 {} + 2>/dev/null || true
    fi
}
_harden_perms

# Restrict umask for the rest of this process tree so any *new* file the
# server creates (rotated logs, retry-queue WAL, scheduler state, etc.)
# is born 600 instead of 644. This is the half of F-07 / F-08 that
# `chmod` cannot fix on its own — it covers files that don't yet exist.
umask 077

# 1. Load base .env (non-secret config)
set -a
source "$SCRIPT_DIR/.env"
set +a

# 2. Decrypt and load secrets
if [[ -f "$SECRETS_ENC" ]]; then
    if [[ ! -f "$AGE_KEY" ]]; then
        echo "ERROR: age key not found at $AGE_KEY" >&2
        exit 1
    fi
    echo "Decrypting secrets from $SECRETS_ENC ..."
    eval "$("$AGE_BIN" -d -i "$AGE_KEY" "$SECRETS_ENC" | grep -v '^#' | grep '=' | sed 's/^/export /')"
    echo "Secrets loaded."
else
    echo "WARNING: No encrypted secrets file at $SECRETS_ENC — using env as-is" >&2
fi

# 3. Start server
echo "Starting MemOS on port $PORT ..."
exec python3.12 -m memos.api.server_api --port "$PORT"
