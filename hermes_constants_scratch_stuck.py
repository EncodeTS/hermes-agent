"""Scratch entries the prune cannot delete as this user: remembered once, skipped until they change.

A container that ran as root and wrote into a bind mount under scratch (the Docker terminal
backend without ``docker_run_as_host_user``, a test or review lane's ``docker run -v``) leaves
files this user cannot unlink. ``rmtree(ignore_errors=True)`` removes what it can, the entry
stays idle, and every hourly pass re-walked the whole subtree and logged the same
"could not fully remove" line again (218 times per entry on one host). Here such an entry is
recorded with a fingerprint of its top directory, the user gets ONE line naming the command
that clears it, and later passes skip it before walking it until the fingerprint changes or
the entry is gone. Nothing is escalated or shelled out to: the hint is for the user to run.

POSIX only: without ``os.getuid`` (Windows) a PermissionError is usually a transient open
handle, so the prune keeps retrying there exactly as before.
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

STUCK_STATE_NAME = ".prune_stuck.json"
_PERMISSION_ERRNOS = frozenset({errno.EACCES, errno.EPERM})


def tracking_supported() -> bool:
    return os.name == "posix" and hasattr(os, "getuid")


def _fingerprint(entry: Path) -> list[int] | None:
    try:
        st = os.lstat(entry)
    except OSError:
        return None
    # ctime too: ``sudo chown -R``/``chmod -R`` fixes the entry without touching its mtime.
    return [st.st_ino, st.st_mtime_ns, st.st_ctime_ns]


def _state_path(root: Path) -> Path:
    return root / STUCK_STATE_NAME


def _read(root: Path) -> dict[str, dict]:
    try:
        data = json.loads(_state_path(root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write(root: Path, state: dict[str, dict]) -> None:
    path = _state_path(root)
    with contextlib.suppress(OSError):
        if not state:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def load_skips(root: Path, *, persist: bool = True) -> tuple[dict[str, dict], frozenset[str]]:
    """``(state, names to skip)``. Records whose entry vanished or whose fingerprint changed
    (someone touched or partly cleaned it) are dropped, so that entry is judged normally."""
    if not tracking_supported():
        return {}, frozenset()
    state = _read(root)
    live = {name: rec for name, rec in state.items() if rec.get("fp") == _fingerprint(root / name)}
    if persist and live != state:
        _write(root, live)
    return live, frozenset(live)


def rmtree_collecting_permission_errors(entry: Path) -> bool:
    """``rmtree`` that keeps going past errors (like ``ignore_errors=True``) and returns True
    when any of them was EACCES/EPERM, i.e. what is left is not removable by this user."""
    denied = False

    def _onexc(_func, _path, exc):
        nonlocal denied
        if isinstance(exc, OSError) and exc.errno in _PERMISSION_ERRNOS:
            denied = True

    if sys.version_info >= (3, 12):
        shutil.rmtree(entry, onexc=_onexc)
    else:  # pragma: no cover - 3.11
        shutil.rmtree(entry, onerror=lambda f, p, info: _onexc(f, p, info[1]))
    return denied


def removal_hint(entry: Path) -> str:
    hint = f"sudo rm -rf {shlex.quote(os.fspath(entry))}"
    if shutil.which("docker"):
        hint += (
            f"  (or without sudo: docker run --rm -v {shlex.quote(os.fspath(entry.parent) + ':/s')}"
            f" alpine rm -rf {shlex.quote('/s/' + entry.name)})"
        )
    return hint


def record(root: Path, state: dict[str, dict], entry: Path, audit) -> None:
    """Remember *entry* as not removable by this user and log the one actionable line."""
    fp = _fingerprint(entry)
    if fp is None:
        return
    state[entry.name] = {"fp": fp, "first_seen": time.time()}
    _write(root, state)
    audit.info(
        "scratch prune: cannot remove %r: it holds files this user (uid %d) may not delete, "
        "usually root-owned output of a Docker container; skipping it until it changes. "
        "Remove it with: %s",
        os.fspath(entry), os.getuid(), removal_hint(entry),  # windows-footgun: ok — only reached when tracking_supported()
    )


def stuck_entries(root: Path) -> list[Path]:
    """Entries the prune is currently skipping as not removable (for ``hermes doctor``)."""
    _state, names = load_skips(root, persist=False)
    return [root / name for name in sorted(names)]
