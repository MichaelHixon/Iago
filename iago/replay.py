"""Record a live run's replies, then replay them with no model at all.

A stranger who clones the repo has no Ollama and no API key. `--target replay` lets them run the
real loop (runner, judge, report) against replies recorded from a live model, so the first run
shows a real verdict in seconds. The replay never invents a reply: a call the fixture does not hold
raises, and the runner records that trial as an error naming the re-record command.

Fixture format (JSONL):
    line 1:  {"record": "replay-header", "format": 1, "recorded_model", "recorded_at",
              "recorded_is_local", "unsupported_controls", "iago_version",
              "technique_library_sha256", "category"}
    then:    {"key": <sha256>, "replies": [{"text": str, "truncated": bool}, ...]}

The key is sha256 over the call's message list, seed and temperature, so a replay honors both
controls by requiring the recorded values. Even with the seed in the key, two calls can share one:
the determinism probe sends the same prompt twice at the same seed and compares. So each key holds
an ordered list of replies, served in call order; a call past the end is a miss, never a repeat.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import dropped_controls, sha256_text
from .target import Target, TruncatedReply, single_turn_messages

REPLAY_HEADER = "replay-header"
REPLAY_FORMAT = 1
#: The fixture `--target replay` loads when `--model` is left on its default.
DEFAULT_REPLAY_FIXTURE = Path(__file__).resolve().parent.parent / "samples" / "replay" / "prompt-extraction.jsonl"
#: The sampling controls a replay can honor, by keying on them.
_KEYED_CONTROLS = frozenset({"seed", "temperature"})


def replay_key(messages: list[dict], options: dict | None) -> str:
    """sha256 over the canonical message list plus seed and temperature. Token caps stay out: the
    runner never sends one prompt twice with only the cap changed."""
    opts = options or {}
    return sha256_text(json.dumps(
        {"messages": [{"role": m["role"], "content": m["content"]} for m in messages],
         "seed": opts.get("seed"), "temperature": opts.get("temperature")},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _valid_row(row: object) -> bool:
    """A fixture row has a string key and a non-empty list of replies, each with string text."""
    return (isinstance(row, dict) and isinstance(row.get("key"), str)
            and isinstance(row.get("replies"), list) and bool(row["replies"])
            and all(isinstance(r, dict) and isinstance(r.get("text"), str) for r in row["replies"]))


class RecordingTarget(Target):
    """Wraps a live target, passes every call through unchanged, and keeps each reply under its
    replay key. It wraps the RAW target (a guard goes outside it), so a guard's block sentinel is
    never stored as a model reply. `save()` writes the fixture after the run."""

    def __init__(self, inner: Target) -> None:
        self.inner = inner
        self.name = inner.name
        self.is_local = inner.is_local
        self.supported_options = inner.supported_options
        self._replies: dict[str, list[dict]] = {}

    def _keep(self, messages: list[dict], options: dict | None, reply: str) -> str:
        self._replies.setdefault(replay_key(messages, options), []).append(
            {"text": str(reply), "truncated": isinstance(reply, TruncatedReply)})
        return reply

    def generate(self, prompt: str, system: str | None = None, options: dict | None = None) -> str:
        reply = self.inner.generate(prompt, system=system, options=options)
        return self._keep(single_turn_messages(prompt, system), options, reply)

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        reply = self.inner.chat(messages, options=options)
        return self._keep(messages, options, reply)

    def save(self, path: Path, manifest: dict | None = None) -> Path:
        """Write the fixture. The header names what was recorded and never a path or host."""
        from . import __version__

        m = manifest or {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"record": REPLAY_HEADER, "format": REPLAY_FORMAT, "recorded_model": self.name,
                  "recorded_at": m.get("created") or datetime.now(timezone.utc).isoformat(),
                  "recorded_is_local": self.is_local,
                  "unsupported_controls": dropped_controls(m),
                  "iago_version": __version__,
                  "technique_library_sha256": m.get("technique_library_sha256"),
                  "category": m.get("category")}
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for key in sorted(self._replies):
                fh.write(json.dumps({"key": key, "replies": self._replies[key]}, ensure_ascii=False) + "\n")
        return path


class ReplayTarget(Target):
    """Answers from a recorded fixture. Local (nothing leaves the machine); it drops exactly the
    sampling controls the recorded backend dropped, so the generic dropped-control disclosure tells
    the recorded truth; and it says what it is in its name and in `manifest_extra`, so no report
    can read as a live measurement."""

    is_local = True

    def __init__(self, fixture: str | Path | None = None) -> None:
        self.path = Path(fixture) if fixture else DEFAULT_REPLAY_FIXTURE
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"replay fixture not readable: {self.path} ({exc.strerror or exc})") from None
        lines = [ln for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        try:
            header = json.loads(lines[0]) if lines else {}
            rows = [json.loads(ln) for ln in lines[1:]]
        except json.JSONDecodeError as exc:
            raise ValueError(f"replay fixture {self.path.name} is not valid JSONL: {exc}") from None
        if (not isinstance(header, dict) or header.get("record") != REPLAY_HEADER
                or header.get("format") != REPLAY_FORMAT):
            raise ValueError(f"replay fixture {self.path.name} has no format-{REPLAY_FORMAT} replay header")
        bad = next((n for n, r in enumerate(rows, 2) if not _valid_row(r)), None)
        if bad is not None:
            raise ValueError(f"replay fixture {self.path.name} line {bad} is not a "
                             '{"key": str, "replies": [{"text": str, ...}, ...]} row')
        self.replies: dict[str, list[dict]] = {r["key"]: r["replies"] for r in rows}
        self.supported_options = _KEYED_CONTROLS - set(header.get("unsupported_controls") or [])
        self._served: dict[str, int] = {}
        self.name = f"replay:{header.get('recorded_model')}"
        self.manifest_extra = {"replay": {
            "fixture": self.path.name,
            "fixture_sha256": hashlib.sha256(raw).hexdigest(),
            "recorded_model": header.get("recorded_model"),
            "recorded_at": header.get("recorded_at"),
            "recorded_is_local": header.get("recorded_is_local", True),
            "recorded_iago_version": header.get("iago_version"),
            "recorded_technique_library_sha256": header.get("technique_library_sha256"),
        }}

    def _answer(self, messages: list[dict], options: dict | None) -> str:
        key = replay_key(messages, options)
        replies = self.replies.get(key, [])
        i = self._served.get(key, 0)
        if i >= len(replies):
            raise RuntimeError(
                f"replay miss: fixture {self.path.name} holds no reply for this call (the prompts, "
                "seed, temperature or call count differ from the recorded run). Re-record it live "
                "with `iago run --record-replay <path>` using the same flags.")
        self._served[key] = i + 1
        rec = replies[i]
        return TruncatedReply(rec["text"]) if rec.get("truncated") else rec["text"]

    def generate(self, prompt: str, system: str | None = None, options: dict | None = None) -> str:
        return self._answer(single_turn_messages(prompt, system), options)

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        return self._answer(messages, options)
