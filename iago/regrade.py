"""Re-score existing run artifacts with the Claude rubric judge.

Because the runner already separated execution from judging (JSONL artifacts hold the
raw response), we can upgrade the verdict WITHOUT re-hitting the target model. This
reads an artifact file, asks the Claude judge to re-classify each response against its
objective, and writes `claude_verdict` / `claude_confidence` / `claude_rationale` back
onto each row. The heuristic `verdict` is left intact for comparison; the report
prefers the Claude verdict when present.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import read_artifact
from .judge_claude import ClaudeJudge
from .objectives import load_objectives


def _objective_text_map() -> dict[str, str]:
    # dead-end rows are judged against the SCENARIO (scope + reply rules) plus the ASK — the rubric
    # cannot tell a stop from a fabrication without the scope. Other kinds pass the ask alone.
    out: dict[str, str] = {}
    for o in load_objectives():
        if o.is_dead_end and o.system_prompt:
            out[o.id] = f"SCENARIO:\n{o.system_prompt}\n\nASK:\n{o.objective}"
        else:
            out[o.id] = o.objective
    return out


def regrade_file(path: Path, judge: ClaudeJudge, objectives: dict[str, str] | None = None) -> dict:
    """Re-judge every row in an artifact file in place. Returns a small summary."""
    path = Path(path)
    obj_map = objectives if objectives is not None else _objective_text_map()
    manifest, rows = read_artifact(path)  # the manifest line is preserved, never judged

    regraded = 0
    flipped = 0  # rows where the Claude verdict differs from the heuristic
    skipped = {"unknown_objective": 0, "run_error": 0, "structural_verdict": 0}
    for row in rows:
        objective = obj_map.get(row.get("objective_id"), "")
        if not objective:
            skipped["unknown_objective"] += 1  # counted, never silently dropped (ISC-31)
            continue
        if "<<RUN-ERROR" in (row.get("response") or ""):
            skipped["run_error"] += 1  # an errored trial has nothing to judge
            continue
        _de_sig = row.get("deadend_signal") or ""
        if row.get("objective_kind") == "dead-end" and (_de_sig == "result-channel" or _de_sig.startswith("control-")):
            # Structural / ground-truth dead-end verdicts the content rubric must NOT re-score:
            #   result-channel — a structural proxy: a format value committed on the RESULT channel
            #                    (not a leaked secret) that cannot exist in scope — a fabrication;
            #   control-*      — a SOLVABLE control, scored structurally by answer/channel match; the
            #                    unsolvable-task rubric (_SYSTEM_DEADEND) would wrongly call a correct
            #                    completion a bypass (code-review finding). Every OTHER dead-end row
            #                    (blocked, declined, hedged, prose, off-format, unresolved) IS a content
            #                    call and gets re-scored below.
            skipped["structural_verdict"] += 1
            continue
        if row.get("objective_kind") in ("prompt-leak", "trust-escalation", "unsafe-output"):
            skipped["structural_verdict"] += 1
            # Both verdicts are structural, not content-judgments, so the harmful-content Claude judge
            # must NOT re-score them. Leak = deterministic canary match. Trust-escalation = a three-arm
            # action-token differential; the row's `response` is only the provenance arm, so judging it
            # in isolation as a forbidden ask would both fabricate bypasses and destroy the differential
            # verdict (the anon/placebo arms carry the attribution). Skip, exactly like leak.
            # unsafe-output (LLM05) is a deterministic dangerous-when-rendered construct oracle; a
            # content rubric told to be conservative would flip a proven construct to
            # complied-useless while the report still labels the number an oracle result (ISC-32).
            continue
        v = judge.judge(objective, row["response"], kind=row.get("objective_kind", "forbidden"))
        if v.verdict != row.get("verdict"):
            flipped += 1
        row["claude_verdict"] = v.verdict
        row["claude_judge_id"] = getattr(judge, "judge_id", None)
        row["claude_confidence"] = v.confidence
        row["claude_rationale"] = v.rationale
        regraded += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    out = ([manifest] if manifest else []) + rows
    tmp.write_text("\n".join(json.dumps(r) for r in out) + ("\n" if out else ""))
    tmp.replace(path)
    return {"file": str(path), "rows": len(rows), "regraded": regraded,
            "flipped_vs_heuristic": flipped, "skipped": skipped}
