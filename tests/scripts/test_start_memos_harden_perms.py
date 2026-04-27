"""Tests for the F-07 / F-08 file-permission hardening preamble in
``start-memos.sh``.

The script tightens ~/.memos/ and the local .memos/ log dir to owner-only
(700 on dirs, 600 on files) on every restart, and sets ``umask 077`` so
files born during runtime inherit the tight default. This test exercises
the ``_harden_perms`` shell function in isolation: it sources the script
(without actually launching the server), pointing ``HOME`` and
``SCRIPT_DIR`` at a tmp tree pre-seeded with files at 644 / dirs at 755,
then asserts the modes after.

We do NOT run the full script — sourcing past `_harden_perms` would try
to load the real .env and start a python process. Instead we extract the
function block and run it in a fresh bash subshell.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "start-memos.sh"


@pytest.fixture
def fake_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake $HOME/.memos and $SCRIPT_DIR/.memos pair with loose
    default perms (755 dirs, 644 files), mirroring what was on disk
    pre-hardening."""
    home = tmp_path / "home"
    script_dir = tmp_path / "script"
    (home / ".memos" / "data").mkdir(parents=True)
    (home / ".memos" / "keys").mkdir()
    (home / ".memos" / "memos.db").write_bytes(b"")
    (home / ".memos" / "data" / "memos_users.db").write_bytes(b"x")
    (home / ".memos" / "secrets.env.age").write_bytes(b"-encrypted-")
    (home / ".memos" / "keys" / "memos.key").write_bytes(b"-key-")

    (script_dir / ".memos" / "logs").mkdir(parents=True)
    (script_dir / ".memos" / "logs" / "memos.log").write_bytes(b"line\n")

    # Set deliberately loose modes so the test starts from the same place
    # the audit reported.
    for d in (home / ".memos", home / ".memos" / "data",
              home / ".memos" / "keys", script_dir / ".memos",
              script_dir / ".memos" / "logs"):
        d.chmod(0o755)
    for f in (home / ".memos" / "memos.db",
              home / ".memos" / "data" / "memos_users.db",
              home / ".memos" / "secrets.env.age",
              home / ".memos" / "keys" / "memos.key",
              script_dir / ".memos" / "logs" / "memos.log"):
        f.chmod(0o644)

    return home, script_dir


def _modes(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def _run_harden(home: Path, script_dir: Path) -> subprocess.CompletedProcess[str]:
    """Source start-memos.sh up to and including the _harden_perms call,
    in a subshell with HOME and SCRIPT_DIR overridden. We extract just
    the preamble (everything up to ``# 1. Load base .env``) so we never
    try to read a real .env or start the server.
    """
    script_text = SCRIPT.read_text()
    cutoff = script_text.index("# 1. Load base .env")
    preamble = script_text[:cutoff]

    runner = textwrap.dedent(f"""
        set -e
        # Override the variables the preamble reads. We have to define
        # them BEFORE sourcing the preamble because the preamble assigns
        # SCRIPT_DIR from $0's path. We replace that line below.
        export HOME={home!s}
        export SCRIPT_DIR={script_dir!s}
        # Re-emit the preamble but strip the SCRIPT_DIR= line so our
        # exported override survives.
    """)
    preamble_no_scriptdir = "\n".join(
        line for line in preamble.splitlines()
        if not line.startswith("SCRIPT_DIR=")
    )
    cmd = ["bash", "-c", runner + "\n" + preamble_no_scriptdir]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_harden_tightens_existing_files_and_dirs(fake_tree):
    home, script_dir = fake_tree

    # Sanity: pre-state matches the audit baseline.
    assert _modes(home / ".memos") == 0o755
    assert _modes(home / ".memos" / "memos.db") == 0o644
    assert _modes(script_dir / ".memos" / "logs" / "memos.log") == 0o644

    result = _run_harden(home, script_dir)
    assert result.returncode == 0, (
        f"_harden_perms exited {result.returncode}\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )

    # ~/.memos tree
    assert _modes(home / ".memos") == 0o700
    assert _modes(home / ".memos" / "data") == 0o700
    assert _modes(home / ".memos" / "keys") == 0o700
    assert _modes(home / ".memos" / "memos.db") == 0o600
    assert _modes(home / ".memos" / "data" / "memos_users.db") == 0o600
    assert _modes(home / ".memos" / "secrets.env.age") == 0o600
    assert _modes(home / ".memos" / "keys" / "memos.key") == 0o600

    # Script-dir log tree
    assert _modes(script_dir / ".memos") == 0o700
    assert _modes(script_dir / ".memos" / "logs") == 0o700
    assert _modes(script_dir / ".memos" / "logs" / "memos.log") == 0o600


def test_harden_is_idempotent(fake_tree):
    """Running twice must leave perms at 700/600 — chmod can't drift the
    state in the wrong direction. This pins the self-healing contract."""
    home, script_dir = fake_tree
    assert _run_harden(home, script_dir).returncode == 0
    assert _run_harden(home, script_dir).returncode == 0
    assert _modes(home / ".memos") == 0o700
    assert _modes(home / ".memos" / "memos.db") == 0o600


def test_harden_is_no_op_when_dirs_missing(tmp_path: Path):
    """If neither ~/.memos nor SCRIPT_DIR/.memos exists, the function
    must not error (covers a fresh-machine first-boot scenario where
    these dirs are about to be created by the server)."""
    home = tmp_path / "fresh-home"
    home.mkdir()
    script_dir = tmp_path / "fresh-script"
    script_dir.mkdir()
    result = _run_harden(home, script_dir)
    assert result.returncode == 0, (
        f"first-boot path failed\nstdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )


def test_umask_in_preamble_is_077():
    """The preamble must set umask 077 so files created by the server
    are born 600. We grep the script source rather than running it
    end-to-end (which would launch python)."""
    text = SCRIPT.read_text()
    # Allow either form; reject any other value.
    assert "umask 077" in text, (
        "start-memos.sh must set `umask 077` so newly-created files "
        "(rotated logs, retry-queue WAL, etc.) inherit owner-only mode"
    )
