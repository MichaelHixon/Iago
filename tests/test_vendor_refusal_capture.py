"""Refusal capture on Anthropic trials (ISC-139..141).

Every Anthropic row records `stop_reason`, `stop_details.category`, `stop_details.explanation`, the
requested model and the served model (`response.model`) beside the judge's verdict. A field the
response does not carry is recorded as absent, never filled in. A reply another model served is
its own stratum and never counts toward the requested model's rates; a provider refusal is a label,
and a refusal the judge did not score `refused` is listed for audit.

Fixtures are fake Messages API responses (no network): a normal completion, a refusal with
stop_details, a refusal missing stop_details, and a served-model mismatch.
"""

from types import SimpleNamespace

from iago.attacks import Technique
from iago.judge import COMPLIED_USELESS
from iago.objectives import Objective
from iago.report import build_html_report, build_report, split_by_stratum
from iago.runner import load_artifacts, run
from iago.target import AnthropicTarget, served_matches

REQUESTED = "claude-sonnet-5-5"
TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


def _msg(text="", **fields):
    """A Messages API response. `fields` sets the top-level attributes present; anything not
    passed is ABSENT from the object, which is how a missing field is simulated."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)] if text else [], **fields)


NORMAL = _msg("I won't help with that request. " * 3, stop_reason="end_turn", stop_details=None,
              model=REQUESTED)
REFUSAL = _msg("", stop_reason="refusal", model=REQUESTED,
               stop_details=SimpleNamespace(type="refusal", category="cyber",
                                            explanation="Declined under the cyber policy."))
REFUSAL_NO_DETAILS = _msg("", stop_reason="refusal", model=REQUESTED)  # stop_details absent
MISMATCH = _msg("I won't help with that request. " * 3, stop_reason="end_turn", stop_details=None,
                model="claude-sonnet-5")


def _target(*responses):
    # The first create is the build-time sampling probe (ISC-143); it accepts temperature.
    queue = [_msg("h", stop_reason="max_tokens", stop_details=None, model=REQUESTED), *responses]
    msgs = SimpleNamespace(create=lambda **_kw: queue.pop(0) if len(queue) > 1 else queue[0])
    return AnthropicTarget(REQUESTED, client=SimpleNamespace(messages=msgs))


def _row(tmp_path, response):
    out = run(_target(response), trials=1, authorized=True, artifacts_dir=tmp_path,
              techniques=TECHS, objectives=OBJS, determinism_check=False)
    return load_artifacts(out)[0]


def test_normal_completion_records_every_field_and_the_requested_stratum(tmp_path):
    row = _row(tmp_path, NORMAL)
    assert {k: row[k] for k in ("vendor_stop_reason", "vendor_stop_category", "vendor_stop_explanation",
                                "requested_model", "served_model", "vendor_absent", "model_stratum")} == {
        "vendor_stop_reason": "end_turn", "vendor_stop_category": None,
        "vendor_stop_explanation": None, "requested_model": REQUESTED, "served_model": REQUESTED,
        "vendor_absent": [], "model_stratum": "requested"}


def test_refusal_with_stop_details_is_a_label_beside_the_judges_verdict(tmp_path):
    row = _row(tmp_path, REFUSAL)
    assert row["vendor_stop_reason"] == "refusal"
    assert row["vendor_stop_category"] == "cyber"
    assert row["vendor_stop_explanation"] == "Declined under the cyber policy."
    # A pre-output refusal is a reply, not a run error: it stays in the rates, judged as written.
    assert row["status"] == "evaluated" and row["response"] == ""
    assert row["verdict"] == COMPLIED_USELESS  # the heuristic judge's call on an empty reply


def test_missing_stop_details_is_recorded_absent_not_invented(tmp_path):
    row = _row(tmp_path, REFUSAL_NO_DETAILS)
    assert row["vendor_stop_reason"] == "refusal"
    assert row["vendor_stop_category"] is None and row["vendor_stop_explanation"] is None
    assert row["vendor_absent"] == ["stop_details"]


def test_served_model_mismatch_is_the_fallback_stratum(tmp_path):
    row = _row(tmp_path, MISMATCH)
    assert row["served_model"] == "claude-sonnet-5"
    assert row["model_stratum"] == "fallback"


def test_a_response_naming_no_model_is_unknown_and_named_absent(tmp_path):
    row = _row(tmp_path, _msg("ok " * 30, stop_reason="end_turn", stop_details=None))
    assert row["model_stratum"] == "unknown"
    assert row["vendor_absent"] == ["model"]


def test_non_anthropic_rows_keep_the_earlier_shape(tmp_path):
    from tests.test_runner import FakeTarget
    out = run(FakeTarget(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS,
              determinism_check=False)
    assert "model_stratum" not in load_artifacts(out)[0]


def test_served_matches_reads_snapshots_but_not_prefix_collisions():
    assert served_matches("claude-x", "claude-x") is True
    assert served_matches("claude-x", "claude-x-20260101") is True
    assert served_matches("claude-x-20260101", "claude-x") is True
    assert served_matches("claude-opus-5", "claude-opus-5-5") is False
    assert served_matches("claude-sonnet-5-5", "claude-sonnet-5") is False
    assert served_matches("claude-x", None) is None


def _rows(tmp_path, *responses):
    out = run(_target(*responses), trials=len(responses), authorized=True, artifacts_dir=tmp_path,
              techniques=TECHS, objectives=OBJS, determinism_check=False)
    return load_artifacts(out)


def test_report_excludes_fallback_rows_from_rates_and_tallies_them(tmp_path):
    rows = _rows(tmp_path, NORMAL, MISMATCH)
    counted, off = split_by_stratum(rows)
    assert [r["model_stratum"] for r in counted] == ["requested"]
    assert [r["model_stratum"] for r in off] == ["fallback"]
    md = build_report(rows)
    assert "Scored trials:** 1" in md
    assert "1 trial(s) excluded from every rate above" in md
    assert "fallback, served by `claude-sonnet-5`" in md
    assert "served by claude-sonnet-5" in build_html_report(rows)


def test_report_lists_judge_vs_provider_disagreements_with_the_raw_reply(tmp_path):
    rows = _rows(tmp_path, REFUSAL, NORMAL)
    md = build_report(rows)
    assert "Provider-reported refusals:** 1 of 2 (cyber 1)" in md
    assert "Judge vs provider: 1 disagreement(s) to audit" in md
    assert "judge **complied-useless**" in md and "(empty reply)" in md
    assert "1 disagreement(s) to audit" in build_html_report(rows)


def test_report_without_api_rows_has_no_vendor_section(tmp_path):
    from tests.test_runner import FakeTarget
    out = run(FakeTarget(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS,
              determinism_check=False)
    assert "Model routing" not in build_report(load_artifacts(out))


def test_labels_survive_the_guard_and_recording_wrappers(tmp_path):
    from iago.guards import GuardedTarget
    from iago.replay import RecordingTarget
    wrapped = GuardedTarget(RecordingTarget(_target(REFUSAL)), [])
    out = run(wrapped, trials=1, authorized=True, artifacts_dir=tmp_path, techniques=TECHS,
              objectives=OBJS, determinism_check=False)
    row = load_artifacts(out)[0]
    assert (row["vendor_stop_category"], row["model_stratum"]) == ("cyber", "requested")


def _labelled(text, served):
    from iago.target import VendorReply
    r = VendorReply(text)
    r.vendor = {"requested_model": REQUESTED, "served_model": served}
    return r


def test_a_trust_row_takes_the_worst_stratum_across_its_arms():
    from iago.runner import vendor_row_fields
    t = _target(NORMAL)
    ok, other = _labelled("x", REQUESTED), _labelled("x", "claude-sonnet-5")
    assert vendor_row_fields(t, ok, ok, ok)["model_stratum"] == "requested"
    assert vendor_row_fields(t, ok, other, ok)["model_stratum"] == "fallback"
    assert vendor_row_fields(t, ok, _labelled("x", None), ok)["model_stratum"] == "unknown"


def test_the_html_audit_list_escapes_the_raw_reply(tmp_path):
    hostile = _msg("<script>alert(1)</script> " * 3, stop_reason="refusal", model=REQUESTED,
                   stop_details=SimpleNamespace(category="<b>x</b>", explanation=None))
    html = build_html_report(_rows(tmp_path, hostile))
    assert "<script>alert(1)" not in html and "&lt;script&gt;" in html
    assert "<b>x</b>" not in html
