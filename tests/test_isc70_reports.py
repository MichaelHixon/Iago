"""ISC-70 — the paired-difference CI reaches the `delta` and `compare` reports, the category
table carries a technique-clustered interval beside the plain Wilson one, and `iago power`
answers the sample-size question. Synthetic rows; no model."""

from __future__ import annotations

import json
from pathlib import Path

from iago.agent_oracle import HIJACKED, RESISTED
from iago.cli import main
from iago.compare import build_comparison, write_comparison_report
from iago.delta import build_delta_report
from iago.judge import BYPASSED, REFUSED
from iago.report import build_report
from iago.stats import newcombe_diff_ci, paired_difference_ci, t_ppf, wilson_interval


# --- delta --------------------------------------------------------------------------------------

def _row(tid, cat, oid, kind, verdict, trial=0, owasp="LLM01"):
    return {"technique_id": tid, "technique_name": f"name-{tid}", "category": cat, "owasp": owasp,
            "objective_id": oid, "objective_kind": kind, "model": "ollama:test", "trial": trial,
            "verdict": verdict, "response": "..." * 20}


def test_delta_reports_a_paired_ci_on_the_leak_reduction():
    # 6 paired prompt-leak trials: a=1 (both leak), b=4 (raw leaked, guard held), c=0, d=1.
    raw, guarded = [], []
    cells = [(BYPASSED, BYPASSED), (BYPASSED, REFUSED), (BYPASSED, REFUSED), (BYPASSED, REFUSED),
             (BYPASSED, REFUSED), (REFUSED, REFUSED)]
    for i, (rv, gv) in enumerate(cells):
        raw.append(_row("t1", "prompt-injection", "obj-leak", "prompt-leak", rv, trial=i, owasp="LLM07"))
        guarded.append(_row("t1", "prompt-injection", "obj-leak", "prompt-leak", gv, trial=i, owasp="LLM07"))
    md = build_delta_report(raw, guarded)
    lo, hi = paired_difference_ci(a=1, b=4, c=0, d=1)   # CI on raw - guarded
    # The report quotes the delta as guarded - raw, so the interval is negated and in points.
    expect = f"{-hi * 100:+.1f} to {-lo * 100:+.1f} pts"
    assert "Newcombe" in md and "paired" in md.lower()
    assert expect in md, (expect, md)
    assert "95% CI on the paired difference" in md


def test_delta_paired_ci_renders_na_when_no_pairs():
    raw = [_row("t1", "role-play", "obj-harm", "forbidden", BYPASSED)]
    guarded = [_row("t2", "role-play", "obj-other", "forbidden", REFUSED)]
    md = build_delta_report(raw, guarded)
    assert "paired difference" in md.lower()
    assert "n/a (0 pairs)" in md


# --- compare ------------------------------------------------------------------------------------

def _artifact(tmp_path: Path, name: str, model: str, attacks: dict[str, list[str]]) -> Path:
    rows = [{"model": model, "kind": "capability", "scenario_id": f"cap{i}", "scenario_name": "cap",
             "verdict": HIJACKED} for i in range(2)]
    for sid, verdicts in attacks.items():
        for t, v in enumerate(verdicts):
            rows.append({"model": model, "kind": "attack", "scenario_id": sid, "scenario_name": sid,
                         "verdict": v, "trial": t})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_compare_reports_an_independent_difference_between_two_models(tmp_path):
    """Trial i on model A and trial i on model B are independent draws — nothing is matched
    across models, whatever the seed — so the interval is Newcombe's UNPAIRED hybrid score and
    the report must not call the design paired, matched or seed-correlated."""
    h, r = HIJACKED, RESISTED
    a = _artifact(tmp_path, "a.jsonl", "modelA", {"sX": [h, h, h, r], "sY": [r, r, r, r]})
    b = _artifact(tmp_path, "b.jsonl", "modelB", {"sX": [r, r, h, r], "sY": [r, r, r, r]})
    comp = build_comparison([a, b])
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert "## Difference between models" in text and "## Paired difference" not in text
    lo, hi = newcombe_diff_ci(3, 4, 1, 4)
    assert f"| sX | 3/4 | 1/4 | +50% | {lo:+.0%} to {hi:+.0%} |" in text
    assert "`modelA` − `modelB`" in text and "independent" in text
    for banned in ("paired", "matched", "same seed", "correlation the"):
        assert banned not in text.split("## Difference between models")[1], banned
    assert "| sY | 0/4 | 0/4 | +0% |" in text


def test_compare_difference_section_absent_with_one_model(tmp_path):
    a = _artifact(tmp_path, "a.jsonl", "modelA", {"sX": [HIJACKED]})
    text = write_comparison_report(build_comparison([a]), reports_dir=tmp_path).read_text()
    assert "## Difference between models" not in text


# --- clustered category interval ----------------------------------------------------------------

def _crow(tid, cat, trial, verdict):
    return dict(technique_id=tid, technique_name=f"n-{tid}", category=cat, objective_id="o1",
                objective_kind="forbidden", model="fake:model", seed=1, temperature=0.8, trial=trial,
                prompt="p", response="A" * 300, verdict=verdict, confidence=0.8, rationale="r",
                latency_s=0.1, timestamp="2026-07-24T00:00:00Z")


def test_category_table_shows_clustered_interval_beside_wilson():
    rows = []
    for tid, hits in (("t1", 4), ("t2", 0), ("t3", 0)):   # the hand-computed deff = 6 case
        for t in range(4):
            rows.append(_crow(tid, "role-play", t, BYPASSED if t < hits else REFUSED))
    rows.append(_crow("t9", "direct-ask", 0, REFUSED))       # one technique -> no cluster estimate
    for tid in ("t7", "t8"):                                 # two techniques, 0% -> no estimate
        rows.append(_crow(tid, "encoding", 0, REFUSED))
    md = build_report(rows)
    cat = md.split("## Bypass Rate by Category")[1].split("## Bypass Rate by Technique")[0]
    assert "95% CI (Wilson)" in cat and "95% CI (technique-clustered)" in cat
    role = next(line for line in cat.splitlines() if line.startswith("| role-play"))
    # Wilson 4/12 = 14%–61%; clustered = Wilson on p = 1/3 at n_eff = 2 with the t(2) = 4.303
    # quantile (deff 6 over 3 techniques), NOT 1.96 -> wider than the 5%–84% the normal gives.
    lo, hi = wilson_interval(2 / 3, 2, z=t_ppf(0.975, 2))
    assert "14%–61%" in role and f"{lo:.0%}–{hi:.0%}" in role and "5%–84%" not in role
    assert "deff 6.0, t(2) over 3 techniques; few clusters, interval is t-widened" in role
    direct = next(line for line in cat.splitlines() if line.startswith("| direct-ask"))
    assert "n/a — 1 cluster: between-technique variance needs >= 2; see Wilson" in direct
    enc = next(line for line in cat.splitlines() if line.startswith("| encoding"))
    assert "n/a — rate at 0%: no between-technique variance; see Wilson" in enc
    assert "deff 1.0" not in enc and "0%–" not in enc.split("| n/a")[1]
    assert "trials within a technique are not independent" in cat
    assert "t(m − 1) quantile" in cat and "Quote the clustered one" not in cat


# --- power helper -------------------------------------------------------------------------------

def test_power_subcommand_prints_connor_sample_size(capsys):
    assert main(["power", "--diff", "0.10", "--discordant", "0.20"]) == 0
    out = capsys.readouterr().out
    assert "155" in out and "Connor" in out and "alpha 0.05" in out and "power 0.80" in out


def test_power_subcommand_floor_when_discordant_omitted(capsys):
    assert main(["power", "--diff", "0.10"]) == 0
    out = capsys.readouterr().out
    assert "77" in out and "lower bound" in out.lower()


def test_power_subcommand_rejects_impossible_inputs(capsys):
    assert main(["power", "--diff", "0.3", "--discordant", "0.2"]) == 2
    assert "ERROR" in capsys.readouterr().err
