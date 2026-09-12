"""ISC-53 — every caveat markdown discloses, HTML discloses too; neither asserts a failure that
did not happen.

#147 closed one header line and its gate closed three sections. These are the six remaining
surfaces the reviewers found, plus the two placement divergences. Each test states the row
combination that produced the divergence, measured before the fix.
"""

import pytest

from iago.report import build_html_report, build_report


def _row(**kw):
    base = dict(
        technique_id="t1", technique_name="Direct", category="direct-ask",
        objective_id="o1", objective_kind="forbidden", model="fake:model",
        seed=1337, temperature=0.8, trial=0, prompt="Do: X", response="A" * 300,
        verdict="bypassed", confidence=0.8, rationale="substantive",
        latency_s=0.1, timestamp="2026-07-24T00:00:00Z",
    )
    base.update(kw)
    return base


def _both(rows):
    return build_report(rows), build_html_report(rows)


def test_partial_forbidden_errors_are_disclosed_in_both_renderers():
    """2 clean trials + 20 errored rendered `0 / 2 … 0% confirmed-bypass rate` in HTML with the
    word "errored" nowhere on the page, while the header still counted 2 forbidden objectives."""
    rows = ([_row(objective_id="o1", trial=i, verdict="refused") for i in range(2)]
            + [_row(objective_id="o2", trial=i, verdict="error") for i in range(20)])
    md, html = _both(rows)
    for doc, name in ((md, "md"), (html, "html")):
        assert "20" in doc and "trial(s) errored" in doc, name
        assert "excluded from every rate above" in doc.lower(), name


def test_all_forbidden_errored_says_nothing_was_measured_in_both_renderers():
    """A 0 / 0 rate with no note reads as a clean bill of health on the copy that gets shared."""
    md, html = _both([_row(objective_id="o1", trial=i, verdict="error") for i in range(3)])
    for doc, name in ((md, "md"), (html, "html")):
        assert "Nothing was measured" in doc, name
        assert "this is NOT a hold" in doc, name


def test_errored_count_spans_the_whole_run_not_just_forbidden():
    """The disclosure claims exclusion from "every rate above", and those rates span five kinds.
    Counting only the forbidden subset reported 5 errored trials as 1."""
    rows = [_row(objective_id="o1", verdict="error")] + [
        _row(objective_id=f"o-{k}", objective_kind=k, verdict="error")
        for k in ("prompt-leak", "trust-escalation", "unsafe-output", "dead-end")]
    md, html = _both(rows)
    for doc, name in ((md, "md"), (html, "html")):
        assert "**5** trial(s) errored" in doc or "<strong>5</strong> trial(s) errored" in doc, name


@pytest.mark.parametrize("errored,expect_note", [(True, True), (False, False)],
                         ids=["unsolvable-errored", "no-unsolvable-rows"])
def test_dead_end_errored_note_only_when_something_errored(errored, expect_note):
    """The gate was `not deadend_unsolvable`, which also empties when the run has no unsolvable
    objectives at all — so a clean controls-only run asserted an error that never happened."""
    rows = [_row(objective_id="o1", verdict="refused"),
            _row(objective_id="od-c", objective_kind="dead-end", verdict="refused",
                 deadend_signal="control-correct")]
    if errored:
        rows.append(_row(objective_id="od-u", objective_kind="dead-end", verdict="error"))
    md, html = _both(rows)
    note = "All unsolvable dead-end trials errored"
    for doc, name in ((md, "md"), (html, "html")):
        assert (note in doc) is expect_note, (name, errored)
        if not expect_note:
            assert "No unsolvable dead-end objectives in this run" in doc, name


def test_control_evidence_survives_an_all_errored_unsolvable_set():
    """`if deadend_controls:` sat inside `if deadend_unsolvable:`, so the one dead-end result that
    DID reach the model was dropped in exactly the run where it was the only evidence."""
    rows = [_row(objective_id="o1", verdict="refused"),
            _row(objective_id="od-u", objective_kind="dead-end", verdict="error"),
            _row(objective_id="od-c", objective_kind="dead-end", verdict="refused",
                 deadend_signal="control-correct")]
    md, html = _both(rows)
    for doc, name in ((md, "md"), (html, "html")):
        assert "Solvable controls" in doc, name
        assert "1 / 1 completed correctly" in doc, name


def test_judge_miscalibration_alarm_reaches_both_renderers():
    """Markdown warns that a control scored `bypassed` means the run's forbidden numbers cannot be
    trusted. HTML had no judge-calibration section at all."""
    md, html = _both([_row(objective_id="o1", verdict="refused"),
                      _row(objective_id="oc", objective_kind="control", verdict="bypassed")])
    for doc, name in ((md, "md"), (html, "html")):
        assert "should not be trusted" in doc, name


def test_decode_caveat_present_in_both_renderers_whether_or_not_decode_failed():
    """HTML disclosed decode FAILURES but not the positive case, so a reader of the shared copy had
    no reason to condition a gated technique's rate on decode success."""
    ok = [_row(objective_id="o1", verdict="refused", gated=True, decoded=True)]
    bad = [_row(objective_id="o1", verdict="refused", gated=True, decoded=False)]
    for doc, name in zip(_both(ok), ("md", "html")):
        assert "conditioned on decode success" in doc, name
    for doc, name in zip(_both(bad), ("md", "html")):
        assert "decode failure" in doc.lower(), name


def test_scorecard_empty_state_matches_across_renderers():
    """HTML emitted a Scorecard heading on an all-errored run; markdown emitted none."""
    md, html = _both([_row(objective_id="o1", verdict="error")])
    for doc, name in ((md, "md"), (html, "html")):
        assert "Scorecard" in doc, name
        assert "No scored objectives in this run" in doc, name


def test_dead_end_method_prose_does_not_precede_its_own_errored_note():
    """HTML rendered three sentences of methodology and THEN said every trial errored."""
    _, html = _both([_row(objective_id="o1", verdict="refused"),
                     _row(objective_id="od", objective_kind="dead-end", verdict="error")])
    assert "All unsolvable dead-end trials errored" in html
    assert "Every other section scores whether the model" not in html


def test_no_reader_facing_string_asserts_the_model_was_unreachable():
    """ERROR is assigned by bare `except Exception` handlers covering 400s, decode errors, rate
    limits and a missing package, so "could not reach the model" was a cause the report cannot know.
    Guards the whole sweep, not one string."""
    from pathlib import Path
    import iago
    src_dir = Path(iago.__file__).parent
    offenders = []
    for path in sorted(src_dir.glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # comments are not shown to a report reader
            if "reach the model" in line or "reached the model" in line:
                offenders.append(f"{path.name}:{i}")
    assert offenders == [], offenders
