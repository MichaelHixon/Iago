"""Review follow-ups for `git.dirty` (ISC-151): paths the first tests could not tell apart from a
silent pass: v4 prefix compression, the stat fast path and racy guard, index extensions and flags,
unmerged entries, a real linked worktree, and the staged-only limit made explicit."""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import pytest

from iago import artifacts
from tests.test_gitdirty import _git, _repo


def _status_paths(root: Path) -> list[str]:
    return sorted(ln[3:] for ln in _git(root, "status", "--porcelain").splitlines())


def test_v4_prefix_compression_with_shared_and_multibyte_prefixes(tmp_path):
    root = _repo(tmp_path, index_version=4)
    long_dir = root / ("d" * 140)  # shared prefix over 127 bytes: a multi-byte offset varint
    for rel in ("sub/a/x1.txt", "sub/a/x2.txt", "sub/b/y.txt"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("one\n")
    long_dir.mkdir()
    for name in ("p1.txt", "p2.txt", "q1.txt"):
        (long_dir / name).write_text("one\n")
    _git(root, "add", "-A")
    _git(root, "update-index", "--index-version", "4")
    _git(root, "commit", "-q", "-m", "prefixes")
    time.sleep(0.02)
    _git(root, "update-index", "--refresh", "-q")
    assert artifacts.git_info(root)["dirty"] is False
    (root / "sub/a/x2.txt").write_text("two\n")
    (long_dir / "p2.txt").write_text("two\n")
    (root / "sub/b/y.txt").unlink()
    info = artifacts.git_info(root)
    assert info["dirty_paths"] == _status_paths(root) == sorted(
        ["sub/a/x2.txt", "sub/b/y.txt", f"{'d' * 140}/p2.txt"]), info


def test_a_clean_tree_is_decided_by_the_stat_fast_path_without_reading_files(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    reads: list[str] = []
    real = Path.read_bytes

    def counting(self):
        reads.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    assert artifacts.git_info(root)["dirty"] is False
    assert [r for r in reads if r != "index"] == []  # nothing hashed: the fast path decided every file


def test_a_file_as_new_as_the_index_is_rehashed_not_trusted(tmp_path, monkeypatch):
    """The racy guard: when the index is not newer than a file, stat equality proves nothing, so the
    content decides. Aging the index to before the files forces every entry through the hash."""
    root = _repo(tmp_path)
    index = root / ".git" / "index"
    old = index.stat().st_mtime - 3600
    os.utime(index, (old, old))
    reads: list[str] = []
    real = Path.read_bytes

    def counting(self):
        reads.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    assert artifacts.git_info(root)["dirty"] is False
    assert {"objectives.yaml", "pyproject.toml", "run.sh"} <= set(reads)


def test_an_unsupported_index_extension_is_unknown_not_clean(tmp_path):
    root = _repo(tmp_path)
    index = root / ".git" / "index"
    data = index.read_bytes()
    index.write_bytes(data[:-20] + b"sdir" + struct.pack(">I", 0) + data[-20:])
    info = artifacts.git_info(root)
    assert info["dirty"] is None and "sdir" in info["dirty_error"], info


def test_skip_worktree_is_ignored_and_intent_to_add_is_dirt_as_git_says(tmp_path):
    root = _repo(tmp_path)
    _git(root, "update-index", "--skip-worktree", "objectives.yaml")  # extended flags: index v3
    (root / "objectives.yaml").write_text("- id: hidden-by-skip-worktree\n")
    (root / "new.txt").write_text("new\n")
    _git(root, "add", "-N", "new.txt")
    info = artifacts.git_info(root)
    assert info["dirty_paths"] == ["new.txt"] == _status_paths(root), info


def test_an_unmerged_path_is_dirt(tmp_path):
    root = _repo(tmp_path)
    base = _git(root, "branch", "--show-current").strip()
    _git(root, "checkout", "-q", "-b", "side")
    (root / "objectives.yaml").write_text("- id: side\n")
    _git(root, "commit", "-q", "-am", "side")
    _git(root, "checkout", "-q", base)
    (root / "objectives.yaml").write_text("- id: base\n")
    _git(root, "commit", "-q", "-am", "base")
    with pytest.raises(Exception):  # the merge conflicts: git exits 1
        _git(root, "merge", "side")
    assert "objectives.yaml" in artifacts.git_info(root)["dirty_paths"]


def test_a_real_linked_worktree_reports_its_own_dirt_not_the_mains(tmp_path):
    root = _repo(tmp_path)
    wt = tmp_path / "linked"
    _git(root, "worktree", "add", "-q", str(wt), "-b", "wt-branch")
    assert artifacts.git_info(wt)["dirty"] is False
    (wt / "objectives.yaml").write_text("- id: edited-in-the-worktree\n")
    info = artifacts.git_info(wt)
    assert info["dirty_paths"] == ["objectives.yaml"] and info["root"] == str(wt), info
    assert artifacts.git_info(root)["dirty"] is False  # the main checkout is untouched


def test_the_staged_only_blind_spot_is_pinned_and_named_in_the_scope(tmp_path):
    """Edit then `git add`: the file on disk equals the index, so tracked-vs-index sees nothing, yet
    the commit the manifest names does not contain it. That is the documented limit; the objectives
    sha256 in the manifest is what still tells such a run apart."""
    root = _repo(tmp_path)
    (root / "objectives.yaml").write_text("- id: staged\n")
    _git(root, "add", "objectives.yaml")
    assert _status_paths(root) == ["objectives.yaml"]  # git itself reports it
    info = artifacts.git_info(root)
    assert info["dirty"] is False  # the limit
    assert "staged" in info["dirty_scope"]
