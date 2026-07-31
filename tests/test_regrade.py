"""ISC-9: regrade re-scores artifacts in place and the report prefers the Claude verdict."""

import json

from iago.judge import BYPASSED, REFUSED
from iago.regrade import regrade_file
from iago.report import build_report


class StubJudge:
    """Returns a fixed verdict — stands in for ClaudeJudge without a client."""

    def __init__(self, verdict, confidence=0.9, rationale="stub"):
        from iago.judge import Verdict
        self._v = Verdict(verdict, confidence, rationale)

    def judge(self, objective, response, kind="forbidden"):
        return self._v


def _write_artifact(tmp_path, rows):
    p = tmp_path / "art.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _row(**kw):
    base = dict(
        technique_id="t1", technique_name="Direct", category="direct-ask",
        objective_id="o1", objective_kind="forbidden", model="m", seed=1,
        temperature=0.8, trial=0, prompt="p", response="A" * 300,
        verdict=BYPASSED, confidence=0.8, rationale="heuristic said bypass",
        latency_s=0.1, timestamp="t",
    )
    base.update(kw)
    return base


def test_regrade_writes_claude_fields_and_counts_flips(tmp_path):
    p = _write_artifact(tmp_path, [_row(verdict=BYPASSED)])
    # Claude says it actually refused → a flip vs the heuristic's "bypassed"
    summary = regrade_file(p, StubJudge(REFUSED), objectives={"o1": "forbidden thing"})
    assert summary["regraded"] == 1
    assert summary["flipped_vs_heuristic"] == 1
    row = json.loads(p.read_text().splitlines()[0])
    assert row["claude_verdict"] == REFUSED
    assert row["verdict"] == BYPASSED  # heuristic left intact for comparison


def test_regrade_skips_run_errors(tmp_path):
    p = _write_artifact(tmp_path, [_row(response="<<RUN-ERROR: boom>>")])
    summary = regrade_file(p, StubJudge(REFUSED), objectives={"o1": "x"})
    assert summary["regraded"] == 0


def test_report_prefers_claude_verdict(tmp_path):
    # Heuristic marked 1/1 bypassed; Claude regrade says refused → report shows 0 bypass.
    rows = [_row(verdict=BYPASSED, claude_verdict=REFUSED, claude_confidence=0.9,
                 claude_rationale="only decoded the request")]
    md = build_report(rows)
    assert "0 / 1 forbidden-objective trials were confirmed bypasses" in md
