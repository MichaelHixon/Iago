"""Which tracked files differ from the git index, read from `.git/index` alone.

`artifacts.git_info` records the checkout's commit by pure file reads, because every agent surface
imports it and those modules carry a first-class no-subprocess, no-socket anti-claim. The commit
alone made a dirty checkout look clean: a run from a tree whose `objectives.yaml` had been edited
named the commit and nothing else. This module closes that without a process, by parsing the index
(versions 2, 3 and 4) and comparing each tracked file with the entry git recorded for it.

SCOPE, stated so a reader cannot take it for more: a path is reported when its working-tree content,
type or executable bit differs from the index entry, when it is missing, or when the index holds a
merge conflict for it. Untracked files and changes that are staged but identical on disk are NOT
seen (the second needs HEAD's tree, which may sit in a pack), and entries marked skip-worktree or
assume-unchanged are skipped, as git skips them. Content is compared as raw bytes and the executable
bit is compared too, so a checkout using `core.autocrlf`, `core.filemode=false` or
`core.symlinks=false` can read as dirty. `git_info` records this scope beside the answer.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from pathlib import Path

DIRTY_SCOPE = ("tracked files whose working-tree content, mode or presence differs from the git "
               "index; not seen: untracked files, staged-only changes, and entries marked "
               "skip-worktree or assume-unchanged (git hides those too); compared as raw bytes and "
               "with the executable bit, so core.autocrlf, core.filemode=false or core.symlinks=false "
               "checkouts can read dirty")
MAX_REPORTED_PATHS = 200

_ASSUME_VALID = 0x8000
_EXTENDED = 0x4000
_STAGE_MASK = 0x3000
_SKIP_WORKTREE = 0x4000  # in the extended flags
_INTENT_TO_ADD = 0x2000  # in the extended flags
_GITLINK = 0o160000
_SYMLINK = 0o120000
_UNSUPPORTED_EXTENSIONS = (b"sdir", b"link")  # sparse index, split index: entries live elsewhere


class IndexUnreadable(ValueError):
    """The index is absent, truncated or of a shape this reader does not parse."""


def _blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # noqa: S324 - git's own object id


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    """git's offset encoding (index v4 path prefix lengths)."""
    c = data[pos]
    pos += 1
    val = c & 127
    while c & 128:
        val += 1
        c = data[pos]
        pos += 1
        val = (val << 7) + (c & 127)
    return val, pos


def _entries(data: bytes):
    """Yield `(path, mode, size, stat_times, sha_hex, stage, skip, intent_to_add)` for each index
    entry; `stat_times` is `(ctime_s, ctime_ns, mtime_s, mtime_ns)`."""
    if len(data) < 32 or data[:4] != b"DIRC":
        raise IndexUnreadable("not a git index")
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3, 4):
        raise IndexUnreadable(f"index version {version} is not supported")
    pos, prev = 12, b""
    for _ in range(count):
        start = pos
        (cs, cns, ms, mns, _dev, _ino, mode, _uid, _gid, size) = struct.unpack(">10I", data[pos:pos + 40])
        sha = data[pos + 40:pos + 60].hex()
        (flags,) = struct.unpack(">H", data[pos + 60:pos + 62])
        pos += 62
        ext = 0
        if version >= 3 and flags & _EXTENDED:
            (ext,) = struct.unpack(">H", data[pos:pos + 2])
            pos += 2
        if version == 4:
            strip, pos = _varint(data, pos)
            end = data.index(b"\0", pos)
            path = prev[:len(prev) - strip] + data[pos:end]
            pos = end + 1
        else:
            end = data.index(b"\0", pos)
            path = data[pos:end]
            pos = start + ((pos - start + len(path) + 8) & ~7)
        prev = path
        skip = bool(flags & _ASSUME_VALID) or bool(ext & _SKIP_WORKTREE)
        yield (os.fsdecode(path), mode, size, (cs, cns, ms, mns), sha, (flags & _STAGE_MASK) >> 12, skip,
               bool(ext & _INTENT_TO_ADD))
    # Extensions follow the entries, then the 20-byte trailer. A sparse or split index keeps part of
    # the file list elsewhere, so a clean answer from the entries alone would be a guess.
    while pos + 8 <= len(data) - 20:
        sig, length = struct.unpack(">4sI", data[pos:pos + 8])
        if sig in _UNSUPPORTED_EXTENSIONS:
            raise IndexUnreadable(f"index extension {sig.decode()!r} is not supported")
        pos += 8 + length


def tracked_changes(root: Path, git_dir: Path) -> list[str]:
    """Paths (relative to `root`) of tracked files that differ from the index at `git_dir/index`.
    Raises `IndexUnreadable` when the index cannot be read or parsed."""
    index = git_dir / "index"
    try:
        data = index.read_bytes()
        index_mtime_ns = index.stat().st_mtime_ns
    except OSError as exc:
        raise IndexUnreadable(f"no readable index ({exc.strerror or exc})") from None
    try:
        entries = list(_entries(data))
    except IndexUnreadable:
        raise
    except (IndexError, ValueError, struct.error) as exc:
        raise IndexUnreadable(f"index could not be parsed ({exc})") from None

    changed: list[str] = []
    for path, mode, size, times, sha, stage, skip, intent in entries:
        if mode == _GITLINK or skip:
            continue
        if stage or intent:  # an unmerged or intent-to-add entry is a pending change by definition
            changed.append(path)
            continue
        full = root / path
        try:
            st = full.lstat()
        except OSError:
            changed.append(path)
            continue
        if mode == _SYMLINK:
            if not stat.S_ISLNK(st.st_mode) or _blob_sha1(os.fsencode(os.readlink(full))) != sha:
                changed.append(path)
            continue
        if not stat.S_ISREG(st.st_mode) or bool(st.st_mode & 0o100) != bool(mode & 0o100):
            changed.append(path)
            continue
        if st.st_size & 0xFFFFFFFF != size:
            changed.append(path)
            continue
        # The stat fast path, as git has it: same ctime and mtime, and older than the index so the
        # file cannot have been rewritten within the index's own timestamp granularity (the "racy"
        # case). Anything else is decided by content.
        if (divmod(st.st_ctime_ns, 1_000_000_000) == times[:2]
                and divmod(st.st_mtime_ns, 1_000_000_000) == times[2:]
                and st.st_mtime_ns < index_mtime_ns):
            continue
        try:
            if _blob_sha1(full.read_bytes()) != sha:
                changed.append(path)
        except OSError:
            changed.append(path)
    return sorted(set(changed))  # an unmerged path has one index entry per stage


def dirty_state(root: Path, git_dir: Path) -> dict:
    """The `git` manifest keys describing working-tree state: `dirty` (True/False, or None with
    `dirty_error` when the index cannot be read — unknown, never "clean"), `dirty_paths` (the
    modified paths, capped, only when dirty) and `dirty_scope`."""
    out: dict = {"dirty": None, "dirty_scope": DIRTY_SCOPE}
    try:
        changed = tracked_changes(root, git_dir)
    except (IndexUnreadable, OSError) as exc:  # an unreadable file or link mid-walk is unknown, not clean
        out["dirty_error"] = str(exc)
        return out
    out["dirty"] = bool(changed)
    if changed:
        out["dirty_paths"] = changed[:MAX_REPORTED_PATHS]
        if len(changed) > MAX_REPORTED_PATHS:
            out["dirty_paths_total"] = len(changed)
    return out
