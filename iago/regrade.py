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

from .judge_claude import ClaudeJudge
from .objectives import load_objectives


def _objective_text_map() -> dict[str, str]:
    return {o.id: o.objective for o in load_objectives()}


def regrade_file(path: Path, judge: ClaudeJudge, objectives: dict[str, str] | None = None) -> dict:
    """Re-judge every row in an artifact file in place. Returns a small summary."""
    path = Path(path)
    obj_map = objectives if objectives is not None else _objective_text_map()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    regraded = 0
    flipped = 0  # rows where the Claude verdict differs from the heuristic
    for row in rows:
        objective = obj_map.get(row.get("objective_id"), "")
        if not objective or "<<RUN-ERROR" in (row.get("response") or ""):
            # Nothing to judge (unknown objective or an errored trial) — skip, don't guess.
            continue
        if row.get("objective_kind") == "prompt-leak":
            # Leak verdict is deterministic (canary match) — the content judge doesn't apply.
            continue
        v = judge.judge(objective, row["response"], kind=row.get("objective_kind", "forbidden"))
        if v.verdict != row.get("verdict"):
            flipped += 1
        row["claude_verdict"] = v.verdict
        row["claude_confidence"] = v.confidence
        row["claude_rationale"] = v.rationale
        regraded += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    tmp.replace(path)
    return {"file": str(path), "rows": len(rows), "regraded": regraded, "flipped_vs_heuristic": flipped}
