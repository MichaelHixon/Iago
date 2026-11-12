"""`git.dirty` in the run manifest (ISC-151): read from `.git/index` by pure file reads.

The manifest used to name the commit and report `dirty: null`, so a run from a tree whose
`objectives.yaml` had been edited read as the stock panel. These tests build real repositories with
the git binary (setup only: the code under test never spawns a process) and compare `git_info`'s
answer with what git itself reports.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from iago import artifacts, gitdirty

_REAL_GIT_INFO = artifacts.git_info

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git binary needed to build fixtures")


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    return subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True,
                          text=True).stdout


def _repo(tmp_path: Path, *, index_version: int | None = None) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    if index_version:
        _git(root, "update-index", "--index-version", str(index_version))
    (root / "pyproject.toml").write_text("[project]\nname = 'iago'\n")
    (root / "objectives.yaml").write_text("- id: a\n")
    (root / "gone.txt").write_text("soon deleted\n")
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("nested\n")
    (root / "run.sh").write_text("#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    os.symlink("objectives.yaml", root / "link")
    _git(root, "add", "-A")
    if index_version:
        _git(root, "update-index", "--index-version", str(index_version))
    _git(root, "commit", "-q", "-m", "init")
    # Let the index timestamp settle past every file's mtime, so the stat fast path is exercised.
    time.sleep(0.02)
    _git(root, "update-index", "--refresh", "-q")
    return root


@pytest.mark.parametrize("version", [2, 3, 4])
def test_clean_checkout_is_false_and_matches_git(tmp_path, version):
    root = _repo(tmp_path, index_version=version)
    assert _git(root, "status", "--porcelain") == ""
    info = artifacts.git_info(root)
    assert info["dirty"] is False and "dirty_paths" not in info, info
    assert info["commit"] == _git(root, "rev-parse", "HEAD").strip()


@pytest.mark.parametrize("version", [2, 4])
def test_editing_a_tracked_file_makes_the_tree_dirty_and_names_the_path(tmp_path, version):
    root = _repo(tmp_path, index_version=version)
    (root / "objectives.yaml").write_text("- id: edited\n")
    info = artifacts.git_info(root)
    assert info["dirty"] is True and info["dirty_paths"] == ["objectives.yaml"], info
    assert "objectives.yaml" in _git(root, "status", "--porcelain")  # git agrees


def test_a_same_size_edit_with_the_old_mtime_is_still_caught_by_content(tmp_path):
    """The stat fast path must not be the only check: a same-length rewrite that restores the old
    mtime changes the content hash and nothing else."""
    root = _repo(tmp_path)
    target = root / "objectives.yaml"
    before = target.stat()
    target.write_text("- id: b\n")  # same length as "- id: a\n"
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert artifacts.git_info(root)["dirty_paths"] == ["objectives.yaml"]


def test_nested_deleted_mode_changed_and_retargeted_paths_are_all_reported(tmp_path):
    root = _repo(tmp_path)
    (root / "sub" / "nested.txt").write_text("changed\n")
    (root / "gone.txt").unlink()
    (root / "run.sh").chmod(0o644)
    (root / "link").unlink()
    os.symlink("elsewhere", root / "link")
    info = artifacts.git_info(root)
    assert info["dirty_paths"] == ["gone.txt", "link", "run.sh", "sub/nested.txt"], info
    assert sorted(ln[3:] for ln in _git(root, "status", "--porcelain").splitlines()) == info["dirty_paths"]


def test_scope_is_recorded_and_untracked_or_ignored_files_are_not_dirt(tmp_path):
    root = _repo(tmp_path)
    (root / "scratch.txt").write_text("untracked\n")
    info = artifacts.git_info(root)
    assert info["dirty"] is False  # documented: untracked files are outside the scope
    assert "untracked" in info["dirty_scope"] and "index" in info["dirty_scope"]


def test_an_unreadable_index_is_unknown_never_clean(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "index").write_bytes(b"not an index at all, just bytes " * 4)
    info = artifacts.git_info(root)
    assert info["dirty"] is None and "not a git index" in info["dirty_error"], info
    (root / ".git" / "index").unlink()
    info = artifacts.git_info(root)
    assert info["dirty"] is None and "no readable index" in info["dirty_error"], info


def test_a_truncated_index_is_unknown_not_a_crash(tmp_path):
    root = _repo(tmp_path)
    data = (root / ".git" / "index").read_bytes()
    (root / ".git" / "index").write_bytes(data[:90])
    assert artifacts.git_info(root)["dirty"] is None


def test_a_long_path_list_is_capped_with_the_true_total(tmp_path):
    root = _repo(tmp_path)
    many = root / "many"
    many.mkdir()
    for i in range(gitdirty.MAX_REPORTED_PATHS + 7):
        (many / f"f{i:04d}.txt").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "many")
    for i in range(gitdirty.MAX_REPORTED_PATHS + 7):
        (many / f"f{i:04d}.txt").write_text("y\n")
    info = artifacts.git_info(root)
    assert info["dirty"] is True
    assert len(info["dirty_paths"]) == gitdirty.MAX_REPORTED_PATHS
    assert info["dirty_paths_total"] == gitdirty.MAX_REPORTED_PATHS + 7


def test_reading_dirt_spawns_no_process(tmp_path, monkeypatch):
    """git_info is on every agent surface's path, and those modules claim no subprocess."""
    root = _repo(tmp_path)
    (root / "objectives.yaml").write_text("- id: edited\n")

    def boom(*a, **k):
        raise AssertionError("git_info spawned a process")

    for name in ("run", "Popen", "check_output", "check_call", "call"):
        monkeypatch.setattr(subprocess, name, boom)
    assert artifacts.git_info(root)["dirty_paths"] == ["objectives.yaml"]


def test_the_manifest_carries_dirty_from_the_checkout_it_ran_from(monkeypatch, tmp_path):
    """build_manifest's `git` block is git_info's, so a run from a dirty tree says so."""
    root = _repo(tmp_path)
    (root / "objectives.yaml").write_text("- id: edited\n")
    monkeypatch.setattr(artifacts, "git_info", lambda: _REAL_GIT_INFO(root))
    manifest = artifacts.build_manifest(surface="chatbot", model="m", sampling={}, judge_id=None)
    assert manifest["git"]["dirty"] is True and manifest["git"]["dirty_paths"] == ["objectives.yaml"]
