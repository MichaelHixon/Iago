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
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import module_fingerprint, sha256_text
from .judge import BYPASSED, HEURISTIC_VERDICTS, NEEDS_REVIEW, judge, judge_leak
from .stats import wilson_interval

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
CONTROL_SET = CALIBRATION_DIR / "judge_control_set.jsonl"
OVERLAY = CALIBRATION_DIR / "judge_control_set.local.jsonl"
METRICS_PATH = CALIBRATION_DIR / "judge_metrics.json"


def _writable_metrics_path() -> Path:
    """Where `judge-eval` stores metrics. The shipped path lives inside the package, which is
    read-only in a wheel install (an uncaught PermissionError) and otherwise mutates the install;
    fall back to the user's config dir (code-review minor)."""
    try:
        if METRICS_PATH.exists() and os.access(METRICS_PATH, os.W_OK):
            return METRICS_PATH
        if os.access(CALIBRATION_DIR, os.W_OK):
            return METRICS_PATH
    except OSError:
        pass
    home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "iago"
    home.mkdir(parents=True, exist_ok=True)
    return home / "judge_metrics.json"

POSITIVE_LABELS = frozenset({"bypassed", "leaked"})
KINDS_FOR = {"heuristic": ("forbidden", "control"), "canary": ("prompt-leak",), "claude": ("forbidden", "control")}

#: Can this judge ever PREDICT the positive class? The heuristic cannot (BYPASSED is not in
#: HEURISTIC_VERDICTS), so its false-positive rate is structurally 0 and its false-negative rate
#: structurally 100% — numbers that describe the verdict vocabulary, not the judge's accuracy.
#: Publishing them as measurements is the failure ISC-35 exists to prevent, so they are reported
#: `n/a` with the reason and the judge is scored on agreement + unresolved rate instead.
POSITIVE_REACHABLE = {"heuristic": BYPASSED in HEURISTIC_VERDICTS, "canary": True, "claude": True}
UNREACHABLE_NOTE = ("this judge never assigns the positive class by design (it is a pre-filter that "
                    "escalates to needs-review), so a false-positive/false-negative rate would "
                    "describe its verdict vocabulary, not its accuracy")


def offline_judge_id() -> str:
    """The id the runner stamps on chatbot rows — heuristic + canary + decode are one scoring stack."""
    return module_fingerprint("judge", "canary", "decode")


def load_control_set(path: Path | str | None = None, overlay: Path | str | None = None) -> list[dict]:
    """The labeled set, with any local overlay bodies merged in by id (never raises on a missing overlay).
    An entry filled from the overlay is tagged `_overlay` so the fingerprint and the report header can
    tell a private measurement from one a clone can reproduce."""
    p = Path(path) if path else CONTROL_SET
    entries = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    op = Path(overlay) if overlay else (p.with_name(p.stem + ".local.jsonl") if path else OVERLAY)
    if op.exists():
        bodies = {json.loads(l)["id"]: json.loads(l)["response"] for l in op.read_text().splitlines() if l.strip()}
        for e in entries:
            if e.get("response") is None and e["id"] in bodies:
                body = bodies[e["id"]]
                want = e.get("response_sha256")
                if want and sha256_text(body) != want:
                    # The withheld entries carry the hash of the exact body they were labeled on;
                    # a mismatched overlay would silently score a different text (code-review minor).
                    raise ValueError(f"overlay body for {e['id']} does not match its recorded "
                                     f"response_sha256 — the label was assigned to different text")
                e["response"] = body
                e["_overlay"] = True
    return entries


def set_fingerprint(entries: list[dict]) -> str:
    """Identity of the set AS SCORED — ids, labels, AND whether each body was available.

    The first version hashed ids and labels only, so the public set and the same set plus the
    gitignored overlay produced the SAME fingerprint while yielding different numbers (91 scored /
    17 positives vs 87 / 13, with every remaining positive benign). The one field whose job is to
    say which set produced a number was blind to the only difference that changes it
    (code-review blocker, ISC-35)."""
    return sha256_text(json.dumps([[e["id"], e["kind"], e["label"], e.get("response") is not None]
                                   for e in entries]))


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n) if n else (0.0, 1.0)
    return {"value": (k / n) if n else None, "lo": lo, "hi": hi, "k": k, "n": n}


def evaluate(judge_name: str, entries: list[dict], *, judge_fn: Callable[[dict], str] | None = None,
             objectives: dict[str, str] | None = None, claude_judge=None,
             positive_reachable: bool | None = None) -> dict:
    """Score one judge over the entries of its kinds. `judge_fn(entry) -> verdict` overrides the
    built-in judge (tests) and, unless `positive_reachable` says otherwise, is assumed able to
    predict the positive class — reachability describes the BUILT-IN judge's vocabulary, and an
    injected fake has its own. Returns the metrics dict (also what `write_metrics` stores)."""
    kinds = KINDS_FOR[judge_name]
    injected = judge_fn is not None   # capture BEFORE the built-in judge is assigned below
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
    positives_by_kind: dict[str, int] = {}
    positives_with_body: dict[str, int] = {}
    scoped = [e for e in entries if e["kind"] in kinds]
    for e in scoped:
        if e["label"] in POSITIVE_LABELS:
            positives_by_kind[e["kind"]] = positives_by_kind.get(e["kind"], 0) + 1
    missing = [e for e in scoped if e.get("response") is None]
    scored = [e for e in scoped if e.get("response") is not None]
    agree = fp = fn = unresolved = 0
    positives = negatives = 0
    disagreements: list[dict] = []
    for e in scored:
        pred = judge_fn(e)
        label_pos = e["label"] in POSITIVE_LABELS
        if label_pos:
            positives_with_body[e["kind"]] = positives_with_body.get(e["kind"], 0) + 1
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
    if positive_reachable is not None:
        reachable = positive_reachable
    elif injected:
        reachable = True          # an injected judge's vocabulary is the caller's, not the built-in's
    else:
        reachable = POSITIVE_REACHABLE.get(judge_name, True)
    if not reachable and (fp or fn != positives):
        # The structural claim is also checked empirically, so a judge that CAN assert the positive
        # class is never silently reported as n/a.
        raise ValueError(f"{judge_name} is declared unable to assign the positive class, but it "
                         f"predicted one ({fp} false positives over {negatives} negatives)")
    return {
        "judge": judge_name, "judge_id": judge_id, "measured": datetime.now(timezone.utc).isoformat(),
        "set": CONTROL_SET.name, "set_sha256": set_fingerprint(entries),
        "set_variant": "with-local-overlay" if any(e.get("_overlay") for e in entries) else "public",
        "n_scored": len(scored), "n_missing_text": len(missing), "positives": positives, "negatives": negatives,
        # WHAT the positives are, not just how many. On this set the scorable positives are benign
        # control-objective compliance; the harmful-bypass positives ship as hashes only, so a public
        # clone measures a harmful FNR of zero-over-zero. A bare "13 positives" implied 13 jailbreaks
        # (cross-vendor audit).
        "positives_by_kind": positives_by_kind, "positives_scored_by_kind": positives_with_body,
        "positive_class_reachable": reachable,
        "agreement": _rate(agree, len(scored)),
        "fpr": _rate(fp, negatives) if reachable else None,
        "fnr": _rate(fn, positives) if reachable else None,
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
    p = Path(path) if path else _writable_metrics_path()
    data = load_metrics(p)
    data.setdefault(result["judge_id"], {})[result["judge"]] = {k: v for k, v in result.items() if k != "disagreements"}
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return p


def _pct(r: dict | None) -> str:
    if r is None or r["value"] is None:
        return "n/a"
    return f"{r['value']:.0%} (95% CI {r['lo']:.0%}–{r['hi']:.0%}, {r['k']}/{r['n']})"


def calibration_line(judge_id: str | None, judge_name: str, metrics: dict | None = None,
                     control_set: Path | str | None = None) -> str:
    """The one-line report-header statement of the judge's measured error rate — or the honest
    absence of one. Never invents a number, and never quotes a number the reader's own copy of the
    control set could not reproduce."""
    data = load_metrics() if metrics is None else metrics
    if not judge_id:
        return (f"**Judge calibration ({judge_name}):** unmeasured — legacy artifact without a `judge_id`; "
                "re-run to stamp one.")
    m = (data.get(judge_id) or {}).get(judge_name)
    if not m:
        return (f"**Judge calibration ({judge_name} `{judge_id}`):** unmeasured for this scoring code — "
                "run `iago judge-eval` to measure agreement / FPR / FNR on the labeled control set.")
    try:
        # Compare against the set AS THE METRICS WERE SCOPED: a row measured with `--no-overlay`
        # must be checked against the public set even on a machine that happens to hold the private
        # overlay, or a correct public measurement would be refused here for the wrong reason.
        overlay = None if m.get("set_variant") == "with-local-overlay" else Path("/nonexistent-overlay")
        here = set_fingerprint(load_control_set(control_set, overlay=overlay))
    except Exception:
        here = None
    if here and m.get("set_sha256") and here != m["set_sha256"]:
        # The stored metrics were measured on a DIFFERENT set than the one on this disk — most often
        # the private overlay of withheld harmful bodies. Quoting them would assert a number the
        # reader cannot reproduce (code-review blocker, ISC-35).
        return (f"**Judge calibration ({judge_name} `{judge_id}`):** stored metrics were measured on a "
                f"different control set than the one in this checkout "
                f"(`{m.get('set_variant', 'unknown')}`); not quoted. Run `iago judge-eval` here to "
                "measure agreement / FPR / FNR on your own copy.")
    missing = f"; {m['n_missing_text']} entr(ies) had no body available" if m.get("n_missing_text") else ""
    by_kind = m.get("positives_scored_by_kind") or {}
    if by_kind:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
        missing += f"; scorable positives are {detail}"
        if m.get("positives_by_kind", {}).get("forbidden", 0) and not by_kind.get("forbidden"):
            missing += (" — every harmful-bypass positive ships as a hash only, so the harmful "
                        "false-negative rate is NOT measured here")
    if m.get("positive_class_reachable") is False:
        return (f"**Judge calibration ({judge_name} `{judge_id}`):** agreement {_pct(m['agreement'])}, "
                f"unresolved {_pct(m['unresolved_rate'])} on {m['n_scored']} reviewer-labeled responses "
                f"({m['positives']} positives{missing}); measured {m['measured'][:10]}. "
                f"False-positive / false-negative rates are **n/a** — {UNREACHABLE_NOTE}. Confirmed "
                "bypasses come from the Claude rubric judge and the deterministic canary oracle, whose "
                "own rates are measured.")
    return (f"**Judge calibration ({judge_name} `{judge_id}`):** agreement {_pct(m['agreement'])}, "
            f"false-positive rate {_pct(m['fpr'])}, false-negative rate {_pct(m['fnr'])}, "
            f"unresolved {_pct(m['unresolved_rate'])} on {m['n_scored']} reviewer-labeled responses "
            f"({m['positives']} positives{missing}); measured {m['measured'][:10]}.")
