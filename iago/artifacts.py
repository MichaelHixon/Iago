"""Artifact provenance + schema (ISC-33).

Every run artifact is a JSONL file whose FIRST line is a manifest record and whose remaining
lines are rows. The manifest is the reproducibility record: what ran (iago version, git commit,
technique-library hash), against what (model tag, Ollama server version, model digest,
quantization, context length), how (every sampling option, the Ollama env knobs that change
batching), on what host, scored by which judge/oracle (`judge_id`, a fingerprint of the scoring
code so a changed rubric can never be silently compared with an old one).

Rows carry `schema_version` and `surface` so a reader can refuse the wrong artifact loudly
(`iago report` on a privilege artifact used to die with a KeyError; `iago compare` on two chatbot
artifacts used to write an empty "no divergence" report). Legacy artifacts (no manifest, no
`surface`) still load: readers infer the surface from the row shape and warn.

Reproducibility is claimed only to the extent this file records it — see README § Reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from . import __version__

SCHEMA_VERSION = 1
MANIFEST_RECORD = "manifest"

#: Ollama server knobs that change batching / KV cache and therefore bit-reproducibility.
OLLAMA_ENV_KEYS = ("OLLAMA_HOST", "OLLAMA_NUM_PARALLEL", "OLLAMA_KV_CACHE_TYPE",
                   "OLLAMA_FLASH_ATTENTION", "OLLAMA_NUM_GPU", "OLLAMA_MAX_LOADED_MODELS")

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


def redact_host(value: str | None) -> str | None:
    """Sanitize OLLAMA_HOST for publication. Artifacts are shared, so the manifest must never carry
    a URL-embedded credential or an internal hostname (cross-vendor audit): userinfo is always
    dropped, a loopback host is kept verbatim because it is the documented target, and anything
    else is recorded as non-local without naming it."""
    if not value:
        return value
    raw = value.strip()
    scheme, sep, rest = raw.partition("://")
    if not sep:
        scheme, rest = "", raw
    rest = rest.split("/", 1)[0]
    if "@" in rest:                       # user:password@host — never published
        rest = rest.rsplit("@", 1)[1]
    host = rest.rsplit(":", 1)[0] if rest.count(":") == 1 else rest
    if host.strip("[]").lower() in _LOCAL_HOSTS:
        return f"{scheme}://{rest}" if scheme else rest
    return "<non-local host redacted>"

_PKG_DIR = Path(__file__).resolve().parent


def sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def module_fingerprint(*modules: str) -> str:
    """A stable id for the scoring code: sha256 over the named iago module sources, in order.
    Any edit to a judge/oracle changes it, which is the point — `compare` refuses to mix runs
    scored by different code unless told to. Returns `<first-module>-<12 hex>`."""
    h = hashlib.sha256()
    for m in modules:
        h.update((_PKG_DIR / f"{m}.py").read_bytes())
        h.update(b"\0")
    return f"{modules[0]}-{h.hexdigest()[:12]}"


def git_info(root: Path | None = None) -> dict:
    """Commit of the checkout iago runs from, read by PURE FILE READS — never a subprocess.

    Provenance is collected by every agent surface, and those modules carry a first-class
    anti-claim that they spawn no process and open no socket. Shelling out to `git` here put a
    `subprocess.run` on that path behind an import the anti-claim's source scans could not see
    (code-review major), so the commit is resolved from `.git` directly: HEAD -> a ref file or a
    packed-refs entry. `dirty` needs a full index comparison and is NOT available without git, so
    it is reported None rather than guessed — a null says "unknown", never "clean".

    Also refuses to report an unrelated ancestor repository's commit: the discovered root must be
    the checkout that actually contains this package's `pyproject.toml`.
    """
    out: dict = {"commit": None, "dirty": None, "root": None}
    start = Path(root) if root else _PKG_DIR.parent
    try:
        for candidate in [start, *start.parents]:
            git_dir = candidate / ".git"
            if not git_dir.exists():
                continue
            if not (candidate / "pyproject.toml").exists():
                break  # a containing repo that is not this package's checkout: not our provenance
            if git_dir.is_file():  # worktree: ".git" is a file pointing at the real dir
                pointer = git_dir.read_text().strip()
                if not pointer.startswith("gitdir: "):
                    break
                git_dir = Path(pointer[len("gitdir: "):])
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                ref = head[len("ref: "):].strip()
                ref_file = git_dir / ref
                if ref_file.exists():
                    out["commit"] = ref_file.read_text().strip()
                else:  # packed refs
                    packed = git_dir / "packed-refs"
                    if packed.exists():
                        for line in packed.read_text().splitlines():
                            if line.endswith(" " + ref):
                                out["commit"] = line.split(" ", 1)[0]
                                break
            elif len(head) == 40:  # detached HEAD
                out["commit"] = head
            out["root"] = str(candidate)
            break
    except Exception:  # unreadable .git — provenance is best-effort, never fatal
        pass
    return out


def ollama_info(model_tag: str | None) -> dict:
    """Server version + the model's digest / quantization / context length, from the LOCAL Ollama
    daemon already in use (never any other host). Every field None when the daemon or model is
    absent — recorded as unknown, never invented."""
    info: dict = {"server_version": None, "model": model_tag, "digest": None,
                  "quantization": None, "parameter_size": None, "context_length": None}
    if not model_tag:
        return info
    tag = model_tag.split(":", 1)[1] if model_tag.startswith("ollama:") else model_tag
    info["model"] = tag
    try:
        import ollama  # local dep; imported lazily so artifact readers never need a daemon

        client = ollama.Client(timeout=5)
        try:
            info["server_version"] = client._client.get("/api/version").json().get("version")
        except Exception:
            pass
        try:
            want = tag if ":" in tag else f"{tag}:latest"
            for m in client.list().models:
                if m.model == want or m.model == tag:
                    info["digest"] = m.digest
                    break
        except Exception:
            pass
        try:
            show = client.show(tag)
            det = show.details
            info["quantization"] = getattr(det, "quantization_level", None)
            info["parameter_size"] = getattr(det, "parameter_size", None)
            mi = show.modelinfo or {}
            ctx = [v for k, v in mi.items() if k.endswith("context_length")]
            info["context_length"] = ctx[0] if ctx else None
        except Exception:
            pass
    except Exception:
        pass
    return info


def scenario_fingerprint(scenarios) -> str | None:
    """sha256 over the scenario objects a surface actually ran, so two agent runs whose manifests
    otherwise match cannot be compared as equals while one used a locally edited scenario YAML
    (cross-vendor audit). Dataclasses are serialized field-wise; anything unserializable falls back
    to repr, which still changes when the content does."""
    from dataclasses import asdict, is_dataclass

    if not scenarios:
        return None
    try:
        payload = [asdict(s) if is_dataclass(s) else repr(s) for s in scenarios]
        return sha256_text(json.dumps(payload, sort_keys=True, default=repr))
    except Exception:
        return sha256_text(repr(scenarios))


def build_manifest(*, surface: str, model: str, sampling: dict, judge_id: str | None,
                   extra: dict | None = None) -> dict:
    """The first JSONL line of every artifact. `sampling` is every option that shapes generation
    (temperature, seeds, trials, step/turn caps, shots); `judge_id` names the scoring code."""
    manifest = {
        "record": MANIFEST_RECORD,
        "schema_version": SCHEMA_VERSION,
        "surface": surface,
        "created": datetime.now(timezone.utc).isoformat(),
        "iago_version": __version__,
        "git": git_info(),
        "model": model,
        "sampling": sampling,
        "judge_id": judge_id,
        # ONLY for an Ollama target. The first gate also matched every non-Ollama tag (`"/" not in
        # model` is true for `gpt-4o`), so an Anthropic run opened three calls to the local daemon
        # and wrote an `ollama` block naming a model Ollama never served (code-review major).
        "ollama": ollama_info(model) if (model or "").startswith("ollama:") else None,
        "ollama_env": {k: (redact_host(os.environ.get(k)) if k == "OLLAMA_HOST" else os.environ.get(k))
                       for k in OLLAMA_ENV_KEYS},
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version()},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(fh: IO[str], manifest: dict) -> None:
    fh.write(json.dumps(manifest) + "\n")
    fh.flush()


def stamp(row: dict, surface: str) -> dict:
    """Add the schema fields every row carries. Mutates and returns `row`."""
    row.setdefault("schema_version", SCHEMA_VERSION)
    row.setdefault("surface", surface)
    return row


def is_manifest(obj: dict) -> bool:
    return isinstance(obj, dict) and obj.get("record") == MANIFEST_RECORD


def read_artifact(path: Path | str) -> tuple[dict | None, list[dict]]:
    """(manifest or None for a legacy file, rows). The manifest is never returned as a row."""
    manifest = None
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if manifest is None and not rows and is_manifest(obj):
            manifest = obj
            continue
        rows.append(obj)
    return manifest, rows


def load_rows(path: Path | str) -> list[dict]:
    """Rows only — the one loader every reader uses, so no reader can mistake the manifest for a
    trial (a manifest row would otherwise KeyError inside every report)."""
    return read_artifact(path)[1]


def surface_of(row: dict) -> str:
    """The row's surface: the stamped field, or inferred from shape for legacy rows."""
    s = row.get("surface")
    if s:
        return s
    if "objective_kind" in row and "technique_id" in row:
        return "chatbot"
    if "outcome" in row and "turns_used" in row:
        return "adaptive"
    if "kind" in row and "scenario_id" in row:
        return "agent"  # a pre-ISC-33 agentic row of unknown surface
    return "unknown"


AGENT_SURFACES = frozenset({"agent", "rag", "a2a", "memory", "misinfo", "disclosure", "privilege", "toolabuse"})


def require_surface(rows: list[dict], expected: str, *, reader: str) -> None:
    """Refuse an artifact of the wrong shape LOUDLY before any KeyError or empty report.
    `expected` is a surface name or the family "agent" (any agentic surface)."""
    if not rows:
        return
    seen = {surface_of(r) for r in rows}
    if expected == "agent":
        bad = {s for s in seen if s not in AGENT_SURFACES}
    else:
        bad = {s for s in seen if s != expected}
    if bad:
        # The remedy is chosen by what the reader FOUND, not by what it wanted: keying it on
        # `expected == "chatbot"` handed chatbot advice to every other surface (cross-vendor audit).
        if "chatbot" in bad:
            remedy = "chatbot `run` artifacts are reported by `iago report`, not by this command"
        else:
            remedy = "agent-surface artifacts are reported by their own `<surface>-run` command or `iago compare`"
        raise ValueError(
            f"{reader} reads {expected} artifacts, but this artifact holds {sorted(seen)} rows — {remedy}"
        )
