"""
Admin Router for Agent Key Management.

Manages agent API keys via the agents-auth.json file that AgentAuthMiddleware reads.
Protected by a dedicated admin key (MEMOS_ADMIN_KEY env var).

Produces v2 hashed entries (bcrypt) compatible with AgentAuthMiddleware.
"""

import hmac
import json
import os
import secrets
from datetime import datetime
from pathlib import Path

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from memos.log import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# Admin auth: a separate env-var key that only the operator knows.
_ADMIN_KEY = os.getenv("MEMOS_ADMIN_KEY", "")

# BCrypt cost factor for newly-created agent keys (F-03). Must be
# >= MIN_BCRYPT_COST (10) — the auth middleware refuses to load any
# hash below that threshold, so generating one here would create a key
# that can't authenticate after the next server restart.
BCRYPT_ROUNDS = 12


def _require_admin(request: Request):
    """Dependency: reject requests without a valid admin key.

    Uses ``hmac.compare_digest`` for the key comparison so the response
    timing does not leak character-by-character match information to a
    network attacker probing for the admin key. Plain ``==`` short-circuits
    on the first mismatching byte and is timing-distinguishable on the
    order of nanoseconds — small but real, and free to fix.

    Header parsing (presence, scheme, split) intentionally short-circuits
    before the comparison: those branches do not depend on the key value
    so they are not a side channel.
    """
    if not _ADMIN_KEY:
        raise HTTPException(status_code=503, detail="Admin API not configured (MEMOS_ADMIN_KEY not set)")
    auth = request.headers.get("Authorization", "").strip()
    if not auth:
        raise HTTPException(status_code=401, detail="Admin key required: Authorization: Bearer <admin-key>")
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid admin key")
    presented = parts[1].strip().encode("utf-8")
    expected = _ADMIN_KEY.encode("utf-8")
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _get_config_path() -> str:
    path = os.getenv("MEMOS_AGENT_AUTH_CONFIG", "")
    if not path:
        raise HTTPException(status_code=503, detail="MEMOS_AGENT_AUTH_CONFIG not set")
    return path


def _read_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"version": 2, "agents": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")


def _write_config(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")


def _hash_key(raw_key: str) -> str:
    """Bcrypt-hash a raw API key.

    Uses ``rounds=BCRYPT_ROUNDS`` explicitly (12 by default, ≥ MIN_BCRYPT_COST
    enforced by AgentAuthMiddleware._load_config). The bcrypt library's
    ``gensalt()`` default is also 12 today, but we pin it here so the hash
    factor is grounded in our own constant rather than the library's
    moving default — and so any future bump (e.g. to 13) is one symbol to
    change. Below MIN_BCRYPT_COST the middleware would reject the hash on
    next startup, so an explicit pin also catches downstream regressions
    early (a 4-rounds hash would never load).
    """
    return bcrypt.hashpw(
        raw_key.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("utf-8")


# --- Request / Response models ---

class CreateAgentKeyRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    description: str = Field("", max_length=500)


class CreateAgentKeyResponse(BaseModel):
    message: str
    key: str
    user_id: str
    description: str


class AgentKeyInfo(BaseModel):
    user_id: str
    key_prefix: str
    description: str


class ListKeysResponse(BaseModel):
    message: str
    agents: list[AgentKeyInfo]


class RevokeKeyRequest(BaseModel):
    user_id: str = Field(..., description="user_id of the agent whose key to revoke")


class SimpleResponse(BaseModel):
    message: str
    success: bool = True


class RotateKeyResponse(BaseModel):
    message: str
    key: str
    user_id: str


# --- Endpoints ---

@router.post(
    "/keys",
    response_model=CreateAgentKeyResponse,
    summary="Create a new agent key",
    dependencies=[Depends(_require_admin)],
)
def create_key(request: Request, body: CreateAgentKeyRequest):
    """Create a new agent API key. The key is only returned once — store it securely."""
    path = _get_config_path()
    config = _read_config(path)
    config.setdefault("version", 2)
    agents = config.get("agents", [])

    if any(a["user_id"] == body.user_id for a in agents):
        raise HTTPException(
            status_code=409,
            detail=f"Agent key for user_id '{body.user_id}' already exists. Use /admin/keys/rotate to replace it.",
        )

    new_key = f"ak_{secrets.token_hex(16)}"
    agents.append({
        "key_hash": _hash_key(new_key),
        "key_prefix": new_key[:12],
        "user_id": body.user_id,
        "description": body.description or f"Agent {body.user_id}",
        "created_at": datetime.utcnow().isoformat(),
    })
    config["agents"] = agents
    _write_config(path, config)

    logger.info(f"[Admin] Created agent key for user_id='{body.user_id}'")
    return CreateAgentKeyResponse(
        message="Agent key created. Store it securely — it won't be shown again.",
        key=new_key,
        user_id=body.user_id,
        description=body.description,
    )


@router.get(
    "/keys",
    response_model=ListKeysResponse,
    summary="List all agent keys",
    dependencies=[Depends(_require_admin)],
)
def list_keys():
    """List all agent keys (prefixes only — full keys are never returned)."""
    path = _get_config_path()
    config = _read_config(path)
    agents = config.get("agents", [])

    return ListKeysResponse(
        message=f"Found {len(agents)} agent key(s)",
        agents=[
            AgentKeyInfo(
                user_id=a["user_id"],
                key_prefix=a.get("key_prefix", "???") + "...",
                description=a.get("description", ""),
            )
            for a in agents
        ],
    )


@router.delete(
    "/keys",
    response_model=SimpleResponse,
    summary="Revoke an agent key",
    dependencies=[Depends(_require_admin)],
)
def revoke_key(request: Request, body: RevokeKeyRequest):
    """Revoke an agent key by user_id. The agent will lose API access immediately."""
    path = _get_config_path()
    config = _read_config(path)
    agents = config.get("agents", [])

    original_count = len(agents)
    agents = [a for a in agents if a["user_id"] != body.user_id]

    if len(agents) == original_count:
        raise HTTPException(status_code=404, detail=f"No agent key found for user_id '{body.user_id}'")

    config["agents"] = agents
    _write_config(path, config)

    logger.info(f"[Admin] Revoked agent key for user_id='{body.user_id}'")
    return SimpleResponse(message=f"Agent key for '{body.user_id}' revoked")


@router.post(
    "/keys/rotate",
    response_model=RotateKeyResponse,
    summary="Rotate an agent key",
    dependencies=[Depends(_require_admin)],
)
def rotate_key(request: Request, body: RevokeKeyRequest):
    """Replace an agent's key with a new one. The old key stops working immediately."""
    path = _get_config_path()
    config = _read_config(path)
    agents = config.get("agents", [])

    found = False
    new_key = f"ak_{secrets.token_hex(16)}"
    for agent in agents:
        if agent["user_id"] == body.user_id:
            agent["key_hash"] = _hash_key(new_key)
            agent["key_prefix"] = new_key[:12]
            agent.pop("key", None)  # Remove any plaintext key from previous rotation
            agent["rotated_at"] = datetime.utcnow().isoformat()
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"No agent key found for user_id '{body.user_id}'")

    _write_config(path, config)

    logger.info(f"[Admin] Rotated agent key for user_id='{body.user_id}'")
    return RotateKeyResponse(
        message="Agent key rotated. Store the new key securely — it won't be shown again.",
        key=new_key,
        user_id=body.user_id,
    )


@router.get(
    "/health",
    summary="Admin health check",
)
def admin_health():
    """Health check for admin endpoints."""
    config_path = os.getenv("MEMOS_AGENT_AUTH_CONFIG", "")
    config_exists = bool(config_path) and Path(config_path).exists()

    return {
        "status": "ok",
        "admin_key_configured": bool(_ADMIN_KEY),
        "auth_config_exists": config_exists,
        "auth_config_path": config_path if config_exists else None,
    }
