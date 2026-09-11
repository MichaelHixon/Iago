"""Judge validation (ISC-35): measure every judge against a reviewer-labeled control set.

A bypass rate is only as credible as the judge that produced it, and a red-team report that
quotes the rate without the judge's own error rate is asking to be trusted. JailbreakBench,
HarmBench, garak and PyRIT all publish judge agreement / FPR / FNR against human labels; this
module does the same for Iago's three judges and stamps the result into every report header.

  heuristic — `judge.judge(response)`, the offline 3-way pre-filter (forbidden + control entries)
  canary    — `judge.judge_leak(canary, response)`, the deterministic LLM07 oracle (prompt-leak)
  claude    — `judge_claude.ClaudeJudge`, the rubric judge (forbidden + control; needs the key)

Positive class = `bypassed` (or `leaked`). FPR = predicted-positive among labeled negatives;
FNR = predicted-non-positive among labeled positives; `needs-review` is a non-positive prediction
and is ALSO reported as the unresolved rate. Every rate carries a 95% Wilson interval. Metrics
are keyed by `judge_id` (a fingerprint of the scoring code), so editing a judge invalidates them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import module_fingerprint, sha256_text
from .judge import BYPASSED, NEEDS_REVIEW, judge, judge_leak
from .stats import wilson_interval

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
CONTROL_SET = CALIBRATION_DIR / "judge_control_set.jsonl"
OVERLAY = CALIBRATION_DIR / "judge_control_set.local.jsonl"
METRICS_PATH = CALIBRATION_DIR / "judge_metrics.json"

POSITIVE_LABELS = frozenset({"bypassed", "leaked"})
KINDS_FOR = {"heuristic": ("forbidden", "control"), "canary": ("prompt-leak",), "claude": ("forbidden", "control")}


def offline_judge_id() -> str:
    """The id the runner stamps on chatbot rows — heuristic + canary + decode are one scoring stack."""
    return module_fingerprint("judge", "canary", "decode")


def load_control_set(path: Path | str | None = None, overlay: Path | str | None = None) -> list[dict]:
    """The labeled set, with any local overlay bodies merged in by id (never raises on a missing overlay)."""
    p = Path(path) if path else CONTROL_SET
    entries = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    op = Path(overlay) if overlay else (p.with_name(p.stem + ".local.jsonl") if path else OVERLAY)
    if op.exists():
        bodies = {json.loads(l)["id"]: json.loads(l)["response"] for l in op.read_text().splitlines() if l.strip()}
        for e in entries:
            if e.get("response") is None and e["id"] in bodies:
                e["response"] = bodies[e["id"]]
    return entries


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n) if n else (0.0, 1.0)
    return {"value": (k / n) if n else None, "lo": lo, "hi": hi, "k": k, "n": n}


def evaluate(judge_name: str, entries: list[dict], *, judge_fn: Callable[[dict], str] | None = None,
             objectives: dict[str, str] | None = None, claude_judge=None) -> dict:
    """Score one judge over the entries of its kinds. `judge_fn(entry) -> verdict` overrides the
    built-in judge (tests). Returns the metrics dict (also what `write_metrics` stores)."""
    kinds = KINDS_FOR[judge_name]
    if judge_fn is None:
        if judge_name == "heuristic":
            judge_fn = lambda e: judge(e["response"]).verdict
        elif judge_name == "canary":
            judge_fn = lambda e: judge_leak(e.get("canary"), e["response"]).verdict
        elif judge_name == "claude":
            if claude_judge is None:
                from .judge_claude import ClaudeJudge
                claude_judge = ClaudeJudge()
            objs = objectives if objectives is not None else _objective_texts()
            judge_fn = lambda e: claude_judge.judge(objs.get(e["objective_id"], e["objective_id"]),
                                                    e["response"], kind=e["kind"]).verdict
        else:
            raise ValueError(f"unknown judge {judge_name!r}")
    scoped = [e for e in entries if e["kind"] in kinds]
    missing = [e for e in scoped if e.get("response") is None]
    scored = [e for e in scoped if e.get("response") is not None]
    agree = fp = fn = unresolved = 0
    positives = negatives = 0
    disagreements: list[dict] = []
    for e in scored:
        pred = judge_fn(e)
        label_pos = e["label"] in POSITIVE_LABELS
        pred_pos = pred == BYPASSED
        positives += label_pos
        negatives += not label_pos
        if pred == NEEDS_REVIEW:
            unresolved += 1
        exact = (pred_pos == label_pos) if e["kind"] == "prompt-leak" else (pred == e["label"])
        if exact:
            agree += 1
        else:
            disagreements.append({"id": e["id"], "label": e["label"], "predicted": pred})
        if pred_pos and not label_pos:
            fp += 1
        if label_pos and not pred_pos:
            fn += 1
    judge_id = claude_judge.judge_id if judge_name == "claude" and claude_judge is not None else offline_judge_id()
    return {
        "judge": judge_name, "judge_id": judge_id, "measured": datetime.now(timezone.utc).isoformat(),
        "set": CONTROL_SET.name, "set_sha256": sha256_text(json.dumps([e["id"] + e["label"] for e in entries])),
        "n_scored": len(scored), "n_missing_text": len(missing), "positives": positives, "negatives": negatives,
        "agreement": _rate(agree, len(scored)), "fpr": _rate(fp, negatives), "fnr": _rate(fn, positives),
        "unresolved_rate": _rate(unresolved, len(scored)), "disagreements": disagreements,
    }


def _objective_texts() -> dict[str, str]:
    from .objectives import load_objectives
    return {o.id: o.objective for o in load_objectives()}


def load_metrics(path: Path | str | None = None) -> dict:
    p = Path(path) if path else METRICS_PATH
    return json.loads(p.read_text()) if p.exists() else {}


def write_metrics(result: dict, path: Path | str | None = None) -> Path:
    """Store under metrics[judge_id][judge_name]; other judges' entries are kept."""
    p = Path(path) if path else METRICS_PATH
    data = load_metrics(p)
    data.setdefault(result["judge_id"], {})[result["judge"]] = {k: v for k, v in result.items() if k != "disagreements"}
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return p


def _pct(r: dict) -> str:
    if r["value"] is None:
        return "n/a"
    return f"{r['value']:.0%} (95% CI {r['lo']:.0%}–{r['hi']:.0%}, {r['k']}/{r['n']})"


def calibration_line(judge_id: str | None, judge_name: str, metrics: dict | None = None) -> str:
    """The one-line report-header statement of the judge's measured error rate — or the honest
    absence of one. Never invents a number."""
    data = load_metrics() if metrics is None else metrics
    if not judge_id:
        return (f"**Judge calibration ({judge_name}):** unmeasured — legacy artifact without a `judge_id`; "
                "re-run to stamp one.")
    m = (data.get(judge_id) or {}).get(judge_name)
    if not m:
        return (f"**Judge calibration ({judge_name} `{judge_id}`):** unmeasured for this scoring code — "
                "run `iago judge-eval` to measure agreement / FPR / FNR on the labeled control set.")
    missing = f"; {m['n_missing_text']} positive(s) had no body available" if m.get("n_missing_text") else ""
    return (f"**Judge calibration ({judge_name} `{judge_id}`):** agreement {_pct(m['agreement'])}, "
            f"false-positive rate {_pct(m['fpr'])}, false-negative rate {_pct(m['fnr'])}, "
            f"unresolved {_pct(m['unresolved_rate'])} on {m['n_scored']} reviewer-labeled responses "
            f"({m['positives']} positives{missing}); measured {m['measured'][:10]}.")
