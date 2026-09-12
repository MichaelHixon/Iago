"""Regression tests for the pre-push code-review findings on ISC-31..35.

Each test pins a number or a sentence that was WRONG in a way that flattered the tool:
a structurally-impossible error rate published as a measurement, metrics quoted from a set the
reader cannot reproduce, an exclusion applied without disclosure, and a run that measured nothing
still exiting 0.
"""

import json
import types

import pytest

from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.judge import BYPASSED, HEURISTIC_VERDICTS, REFUSED, judge
from iago.judge_eval import (CONTROL_SET, METRICS_SCHEMA, calibration_line, evaluate, load_control_set,
                             set_fingerprint, write_metrics)
from iago.objectives import Objective
from iago.report import build_html_report, build_report
from iago.runner import load_artifacts, run
from iago.target import Target

MISSING_OVERLAY = pathlib_missing = __import__("pathlib").Path("/nonexistent-overlay.jsonl")


# --- BLOCKER 1: an unreachable positive class is n/a, never 0% / 100% -------------------------

def test_heuristic_cannot_assign_the_positive_class():
    assert BYPASSED not in HEURISTIC_VERDICTS
    # empirical, over the whole shipped set: not one prediction is the positive class
    entries = load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY)
    verdicts = {judge(e["response"]).verdict for e in entries
                if e["kind"] in ("forbidden", "control") and e["response"] is not None}
    assert BYPASSED not in verdicts


def test_heuristic_metrics_report_na_not_zero_and_hundred():
    m = evaluate("heuristic", load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY))
    assert m["positive_class_reachable"] is False
    assert m["fpr"] is None and m["fnr"] is None     # not 0/74 and 17/17 dressed as measurements
    assert m["agreement"]["n"] == m["n_scored"] and m["unresolved_rate"]["value"] > 0


def test_reachable_judges_still_report_real_rates():
    m = evaluate("canary", load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY))
    assert m["positive_class_reachable"] is True
    assert m["fpr"]["n"] > 0 and m["fnr"]["k"] >= 4


def test_declared_unreachable_judge_that_predicts_a_positive_raises():
    entries = [{"id": "p", "kind": "forbidden", "label": "refused", "response": "x"}]
    with pytest.raises(ValueError, match="declared unable to assign the positive class"):
        evaluate("heuristic", entries, judge_fn=lambda e: BYPASSED, positive_reachable=False)


def test_calibration_line_for_unreachable_judge_says_na_with_the_reason(tmp_path):
    m = evaluate("heuristic", load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY))
    metrics = json.loads(write_metrics(m, tmp_path / "m.json").read_text())
    line = calibration_line(m["judge_id"], "heuristic", metrics=metrics)
    assert "**n/a**" in line and "never assigns the positive class by design" in line
    assert "false-positive rate 0%" not in line and "false-negative rate 100%" not in line


# --- BLOCKER 2: the set fingerprint sees body availability; a mismatch is not quoted ------------

def test_fingerprint_distinguishes_public_set_from_overlay_measurement():
    public = load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY)
    withheld = [e for e in public if e["response"] is None]
    assert withheld, "the shipped set must withhold at least one harmful body"
    filled = [dict(e, response="body", _overlay=True) if e["response"] is None else e for e in public]
    assert set_fingerprint(public) != set_fingerprint(filled)
    assert evaluate("heuristic", filled)["set_variant"] == "with-local-overlay"
    assert evaluate("heuristic", public)["set_variant"] == "public"


def test_calibration_line_refuses_metrics_measured_on_a_different_set(tmp_path):
    """The real failure: metrics measured with the private overlay (91 scored / 17 positives)
    quoted to a reader whose set yields 87 / 13, all of them benign."""
    m = evaluate("heuristic", load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY))
    m["set_sha256"] = "0" * 64          # as if measured elsewhere (e.g. with the private overlay)
    m["set_variant"] = "with-local-overlay"
    metrics = json.loads(write_metrics(m, tmp_path / "m.json").read_text())
    line = calibration_line(m["judge_id"], "heuristic", metrics=metrics)
    assert "different control set than the one in this checkout" in line
    assert "agreement" not in line.split("not quoted")[0].replace("Judge calibration", "")


def test_shipped_metrics_are_public_and_reproducible_here():
    """The committed metrics must be ones a clone reproduces — measured with --no-overlay — and
    must be written by THIS measurement build. The shipped file once predated the disclosure keys,
    so every report header quoted a harmful false-negative rate with its own "not measured on
    harmful bypasses" sentence silently skipped (Council blocker). Shape is now asserted, not just
    provenance."""
    from iago.judge_eval import load_metrics
    data = load_metrics()
    assert data, "judge_metrics.json must ship measured metrics"
    public_fp = set_fingerprint(load_control_set(CONTROL_SET, overlay=MISSING_OVERLAY))
    for judge_id, per_judge in data.items():
        for name, m in per_judge.items():
            assert m["set_variant"] == "public", f"{name} metrics were measured with the private overlay"
            assert m["set_sha256"] == public_fp, f"{name} metrics do not match the shipped control set"
            assert m.get("metrics_schema") == METRICS_SCHEMA, f"{name} metrics are shape-stale"
            assert "positives_by_kind" in m and "positives_scored_by_kind" in m


def test_shipped_header_states_the_harmful_rate_is_unmeasured():
    """The disclosure that matters most, asserted on the rendered string rather than on the code
    path that is supposed to produce it."""
    from iago.judge_eval import load_metrics

    data = load_metrics()
    claude = [(jid, m) for jid, per in data.items() for n, m in per.items() if n == "claude"]
    assert claude, "the shipped metrics must include the rubric judge"
    jid, m = claude[0]
    assert m["positives_by_kind"].get("forbidden", 0) > 0
    assert m["positives_scored_by_kind"].get("forbidden", 0) == 0
    line = calibration_line(jid, "claude", metrics=data)
    assert "harmful false-negative rate is NOT measured here" in line


# --- MAJOR 3 + 4: decode-failure disclosure and the exit-0 hole it opened ----------------------

def _row(**kw):
    base = dict(technique_id="b64", technique_name="Base64", category="encoding", objective_id="o1",
                objective_kind="forbidden", model="m", seed=1, temperature=0.8, trial=0, prompt="p",
                response="I can't help with that.", verdict=REFUSED, confidence=0.9, rationale="r",
                latency_s=0.1, timestamp="t")
    base.update(kw)
    return base


def test_all_decode_failed_run_is_not_reported_as_a_transport_error():
    rows = [_row(gated=True, decoded=False), _row(gated=True, decoded=False, trial=1)]
    text = build_report(rows)
    assert "all 2 forbidden-objective trials were DECODE FAILURES" in text
    assert "RUN-ERROR" not in text.split("## Evidence")[1]
    assert "NOT a hold" in text


def test_mixed_error_and_decode_failure_names_both():
    rows = [_row(gated=True, decoded=False), _row(response="<<RUN-ERROR: boom>>", verdict="error", trial=1)]
    text = build_report(rows)
    assert "1 were RUN-ERRORs and 1 were decode failures" in text


def test_html_report_discloses_the_decode_exclusion():
    rows = [_row(verdict=BYPASSED, response="A" * 300), _row(gated=True, decoded=False, trial=1)]
    html = build_html_report(rows)
    assert "Decode-gating" in html and "EXCLUDED from every harmful-content denominator" in html
    assert "1 / 1 cipher / low-resource trials" in html


def test_cmd_report_exits_nonzero_when_every_trial_was_a_decode_failure(tmp_path, monkeypatch, capsys):
    from iago import cli

    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(_row(gated=True, decoded=False, trial=i)) for i in range(3)))
    monkeypatch.setattr(cli, "write_report", lambda rows, **kw: tmp_path / "r.md")
    rc = cli._cmd_report(types.SimpleNamespace(artifact=str(p), log=False, html=False))
    assert rc == 1
    err = capsys.readouterr().err
    assert "0 of 3 trials probed a guardrail" in err and "decode-failure" in err
