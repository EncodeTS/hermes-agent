"""An idle scratch entry this user cannot fully delete is reported once and not re-walked hourly."""

import os
import stat
import time

import pytest

import hermes_constants_scratch as scratch

pytestmark = pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0,
    reason="needs a POSIX non-root user: root deletes through a read-only dir",
)

OLD = time.time() - 48 * 3600


def _undeletable_entry(root):
    """``lane/ro/blob`` with ``ro`` mode 0500: rmtree removes everything else and fails
    on ``blob`` with EACCES, the same residue a root-owned container write leaves."""
    entry = root / "lane"
    ro = entry / "ro"
    ro.mkdir(parents=True)
    (entry / "deletable.txt").write_text("x", encoding="utf-8")
    (ro / "blob").write_text("left by a container", encoding="utf-8")
    for p in (ro / "blob", entry / "deletable.txt", ro, entry):
        os.utime(p, (OLD, OLD))
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return entry, ro


def _walk_spy(monkeypatch):
    walked = []
    real_bytes, real_state = scratch._tree_bytes, scratch.subtree_touch_state
    monkeypatch.setattr(scratch, "_tree_bytes", lambda p: walked.append(p) or real_bytes(p))
    monkeypatch.setattr(scratch, "subtree_touch_state", lambda p, c: walked.append(p) or real_state(p, c))
    monkeypatch.setattr(scratch, "reap_processes_rooted_in", lambda *_a, **_k: 0)
    return walked


def test_undeletable_entry_logged_once_then_skipped_without_walk(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, ro = _undeletable_entry(root)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert entry in walked
        walked.clear()
        # A day later: pass 1's partial delete no longer counts as recent activity, so on
        # main the entry is selected, walked and logged again (and again every hour after).
        later = time.time() + 48 * 3600
        monkeypatch.setattr(scratch.time, "time", lambda: later)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert entry not in walked, "a known-undeletable entry must not be re-walked every pass"
        lines = [ln for ln in log_file.read_text(encoding="utf-8-sig").splitlines() if repr(str(entry)) in ln]
        assert len(lines) == 1, lines
        assert "sudo rm -rf" in lines[0] and str(entry) in lines[0], lines[0]
        assert (ro / "blob").exists()
    finally:
        ro.chmod(0o700)


def test_changed_or_fixed_entry_is_judged_again(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, ro = _undeletable_entry(root)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        # The one line tells the user how to clear it.
        assert f"sudo rm -rf {entry}" in log_file.read_text(encoding="utf-8-sig")
        # The user fixes the permissions; the entry's own mtime changing is what un-skips it.
        ro.chmod(0o700)
        os.utime(entry, (OLD + 60, OLD + 60))
        walked.clear()
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert entry in walked
        assert not entry.exists()
        assert not (root / ".prune_stuck.json").exists()
    finally:
        if ro.exists():
            ro.chmod(0o700)
