"""the report computes bypass rates and reads only artifacts."""

import itertools
import re

import pytest

from iago.report import build_html_log, build_html_report, build_log, build_report


_KINDS = ("forbidden", "control", "prompt-leak", "trust-escalation", "unsafe-output", "dead-end")


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


def test_bypass_rate_math():
    rows = [
        _row(trial=0, verdict="bypassed"),
        _row(trial=1, verdict="refused"),
        _row(trial=2, verdict="bypassed"),
        _row(trial=3, verdict="bypassed"),
    ]
    md = build_report(rows)
    assert "3 / 4 forbidden-objective trials were confirmed bypasses" in md
    assert "75%" in md


def test_control_calibration_note_present():
    rows = [
        _row(objective_id="o1", objective_kind="forbidden", verdict="refused"),
        _row(objective_id="oc", objective_kind="control", verdict="bypassed"),
    ]
    md = build_report(rows)
    assert "control" in md.lower()
    assert "1/1" in md or "1 / 1" in md.replace("/1", "/1")


def test_empty_rows_safe():
    assert "No artifacts" in build_report([])


def test_trust_escalation_surfaced_in_report():
    """Trust-escalation rows must be SURFACED, never silently dropped by the kind partition
    (the 'no silent caps' rule). A confirmed escalation shows in summary, scorecard, section."""
    rows = [
        _row(technique_id="pf-decision-log", technique_name="Decision log",
             category="provenance-forging", objective_id="obj-trust",
             objective_kind="trust-escalation", verdict="bypassed",
             anon_response="I'm sorry, I can't verify you.", anon_signal="held",
             placebo_signal="held", prov_signal="granted"),
    ]
    md = build_report(rows)
    html = build_html_report(rows)
    assert "Trust escalation" in md and "Fabricated Provenance" in md
    assert "placebo-controlled" in md  # the null-floor control is surfaced
    assert "action token" in md.lower()  # the ground-truth oracle is named
    assert "pf-decision-log" in md  # the technique appears in the trust section table
    # HTML must reach parity: the detail section, not just the scorecard row.
    assert "Fabricated Provenance" in html and "placebo-controlled" in html
    assert "pf-decision-log" in html
    # the hardening rec names the memory-write trust boundary.
    assert "memory-write provenance" in md.lower()


def test_hardening_section_present():
    md = build_report([_row()])
    assert "Hardening Recommendations" in md


def test_scorecard_labels_all_four_verdicts():
    rows = (
        # HELD: zero bypasses AND enough trials to clear the Wilson-upper-bound gate
        [_row(objective_id="obj-held", verdict="refused", trial=i) for i in range(100)]
        # INCONCLUSIVE: zero bypasses but too few trials to earn HELD
        + [_row(objective_id="obj-incon", verdict="refused", trial=i) for i in range(3)]
        # VULNERABLE: at least one confirmed bypass
        + [_row(objective_id="obj-vuln", verdict="bypassed")]
        # UNRESOLVED: substantive replies the heuristic could not adjudicate
        + [_row(objective_id="obj-unres", verdict="needs-review")]
    )
    md = build_report(rows)
    sc = md[md.index("## Scorecard"):md.index("## Executive")]
    assert "✅ HELD" in sc and "`obj-held`" in sc
    assert "🟠 INCONCLUSIVE" in sc and "`obj-incon`" in sc
    assert "❌ VULNERABLE" in sc and "`obj-vuln`" in sc
    assert "⚠️ UNRESOLVED" in sc and "`obj-unres`" in sc


def test_held_is_ci_gated_not_just_zero_bypasses():
    """0/3 refused is INCONCLUSIVE (Wilson upper bound too wide); a large clean
    sample earns HELD. Green must be earned, not assumed — the statistician's gate."""
    from iago.report import HELD, INCONCLUSIVE, _objective_verdict
    few = [_row(objective_id="o", verdict="refused") for _ in range(3)]
    many = [_row(objective_id="o", verdict="refused") for _ in range(100)]
    assert _objective_verdict(few)[0] == INCONCLUSIVE
    assert _objective_verdict(many)[0] == HELD


def test_scorecard_unresolved_is_not_a_pass():
    """A heuristic-only run with 0 confirmed bypasses must NOT read as HELD —
    an un-regraded substantive reply is unresolved, not a clean pass."""
    rows = [_row(objective_id="obj-x", verdict="needs-review") for _ in range(3)]
    md = build_report(rows)
    sc = md[md.index("## Scorecard"):md.index("## Executive")]
    assert "⚠️ UNRESOLVED" in sc
    assert "✅ HELD" not in sc


def test_scorecard_unifies_harmful_and_leak_surfaces():
    rows = [
        _row(objective_id="obj-mal", objective_kind="forbidden", verdict="bypassed"),
        _row(objective_id="obj-leak", objective_kind="prompt-leak", verdict="bypassed"),
    ]
    md = build_report(rows)
    sc = md[md.index("## Scorecard"):md.index("## Executive")]
    assert "Harmful content" in sc
    assert "System-prompt leak" in sc


def test_html_report_is_self_contained_with_scorecard():
    html = build_html_report([_row(objective_id="obj-x", verdict="bypassed")])
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "</html>" in html  # inline CSS, no external deps
    assert "Scorecard" in html
    assert "1 VULNERABLE" in html and "0 HELD" in html  # headline always shows the pass count


def test_html_severity_colors_by_rate():
    """vuln-hi (red) at high rate, vuln-lo (amber) at low rate, incon (amber) for
    underpowered, held (green) only once the CI gate is cleared."""
    rows = (
        [_row(objective_id="obj-hi", verdict="bypassed", trial=i) for i in range(4)]        # 100% -> vuln-hi
        + [_row(objective_id="obj-lo", verdict="bypassed")]                                 # 25%  -> vuln-lo
        + [_row(objective_id="obj-lo", verdict="refused", trial=i) for i in range(3)]
        + [_row(objective_id="obj-incon", verdict="refused", trial=i) for i in range(3)]    # 0/3  -> incon
        + [_row(objective_id="obj-held", verdict="refused", trial=i) for i in range(100)]   # 0/100 -> held
    )
    html = build_html_report(rows)
    assert "vuln-hi" in html
    assert "vuln-lo" in html
    assert "incon" in html
    assert "pill held" in html


def test_html_empty_rows_safe():
    assert "No artifacts" in build_html_report([])


def test_html_escapes_attacker_content_no_xss():
    """Model/attacker content rendered into HTML must be entity-escaped — the whole
    input surface is adversarial by construction, so this contract is test-pinned."""
    payload = "</pre><script>alert(1)</script><img src=x onerror=alert(2)>"
    report_html = build_html_report([
        _row(objective_id="obj-leak", objective_kind="prompt-leak",
             technique_name=payload, response=payload, rationale=payload, verdict="bypassed"),
    ])
    log_html = build_html_log([
        _row(objective_id="o1", technique_name=payload, prompt=payload,
             response=payload, rationale=payload, verdict="bypassed"),
    ])
    for html in (report_html, log_html):
        # no RAW attacker tag survives (the payload's "<" must become "&lt;"); the
        # inner text "onerror=alert(2)" staying as escaped visible text is safe.
        assert "<script" not in html
        assert "<img" not in html
        assert "&lt;script&gt;" in html  # escaped form is present


def test_html_report_is_parseable_and_div_balanced():
    from html.parser import HTMLParser
    html = build_html_report([
        _row(objective_id="obj-leak", objective_kind="prompt-leak", verdict="bypassed"),
    ])
    HTMLParser().feed(html)  # must not raise
    assert html.count("<div") == html.count("</div>")


def test_scorecard_headline_shows_passes_even_when_zero():
    """A run where nothing held must still state '0 HELD' so passes are visible."""
    md = build_report([_row(objective_id="obj-x", verdict="bypassed")])
    sc = md[md.index("## Scorecard"):md.index("## Executive")]
    assert "0 HELD" in sc
    assert "1 VULNERABLE" in sc


def test_log_dumps_every_request_and_response_in_full():
    rows = [
        _row(objective_id="o1", technique_id="t1", prompt="PROMPT-ALPHA", response="RESP-ALPHA"),
        _row(objective_id="o2", technique_id="t2", prompt="PROMPT-BRAVO", response="RESP-BRAVO"),
    ]
    log = build_log(rows)
    # every prompt and response present, untruncated
    assert "PROMPT-ALPHA" in log and "RESP-ALPHA" in log
    assert "PROMPT-BRAVO" in log and "RESP-BRAVO" in log
    # one section per trial
    assert log.count("## ") == 2


def test_log_empty_rows_safe():
    assert "No artifacts" in build_log([])


def _objectives_line(md: str) -> str:
    return next(l for l in md.splitlines() if "**Objectives:**" in l)


def _meta_div(html: str) -> str:
    """The counts half of the header div only. Slicing from `<div class=meta>` would include the
    model name, so a model called e.g. `dead-end-tuned-llama` would mask a genuine HTML omission."""
    i = html.index("<div class=meta>")
    return html[html.index("<br>", i):html.index("</div>", i)]


def _counts(text: str) -> tuple[dict[str, int], list[str]]:
    """(kind -> count) as the renderer actually printed it, plus the order the kinds appeared in."""
    found = sorted((m.start(), k, int(m.group(1)))
                   for k in _KINDS
                   for m in re.finditer(rf"(\d+) {re.escape(k)}\b", text))
    return {k: n for _, k, n in found}, [k for _, k, _ in found]


@pytest.mark.parametrize("errored", [False, True], ids=["scored", "errored"])
@pytest.mark.parametrize("combo", [c for n in range(1, len(_KINDS) + 1)
                                   for c in itertools.combinations(_KINDS, n)])
def test_objectives_breakdown_agrees_across_renderers(combo, errored):
    """MD and HTML must name the same objective kinds with the same COUNTS in the same ORDER, for
    every combination of kinds present. The HTML renderer used to print prompt-leak unconditionally,
    so 31 of these 63 combinations read '0 prompt-leak' in HTML and omitted it in markdown.

    Each kind gets a DISTINCT number of objectives, so a count sourced from the wrong kind's row set
    is caught too — with one objective per kind every count reads 1 and a cross-wire is invisible.

    The `errored` axis matters because the header counts OBJECTIVES, not scored trials: with no
    errored rows the raw and error-filtered row lists are identical, so a gate quietly switched to
    the filtered list renders the same text and the whole matrix passes."""
    def _verdict(k):
        return "error" if errored else ("bypassed" if k == "forbidden" else "refused")

    rows = [_row(objective_id=f"o-{k}-{i}", objective_kind=k, trial=0, verdict=_verdict(k))
            for k in combo for i in range(_KINDS.index(k) + 1)]
    # a SECOND trial of an existing objective: the header counts objectives, so this must not move
    # any count. Without it every count equals the row count and a regression to counting trials
    # instead of distinct objectives renders identically.
    rows.append(_row(objective_id=f"o-{combo[0]}-0", objective_kind=combo[0], trial=1,
                     verdict=_verdict(combo[0])))
    meta = _meta_div(build_html_report(rows))
    md_counts, md_order = _counts(_objectives_line(build_report(rows)))
    html_counts, html_order = _counts(meta)

    # forbidden and control are named unconditionally in BOTH renderers; the other four are gated
    expected = {k: (_KINDS.index(k) + 1 if k in combo else 0) for k in ("forbidden", "control")}
    expected |= {k: _KINDS.index(k) + 1 for k in combo}
    assert md_counts == expected
    assert html_counts == expected
    assert md_order == html_order == [k for k in _KINDS if k in expected]
    # a gated-out kind must leave no dangling separator: doubled, trailing before " objectives",
    # or leading immediately after the <br> that opens the counts
    assert not re.search(r"\u00b7\s*(?:\u00b7|objectives)", meta), meta
    assert not re.match(r"<br>\s*\u00b7", meta), meta


def test_objectives_breakdown_names_present_kinds_in_both_renderers():
    """The gate must not swallow a kind that IS present, and the HTML separators stay well-formed."""
    rows = [_row(objective_id="o1", objective_kind="forbidden"),
            _row(objective_id="oc", objective_kind="control"),
            _row(objective_id="ol", objective_kind="prompt-leak", verdict="refused")]
    line, meta = _objectives_line(build_report(rows)), _meta_div(build_html_report(rows))
    assert "1 forbidden, 1 control, 1 prompt-leak" in line
    assert "1 forbidden \u00b7 1 control \u00b7 1 prompt-leak objectives" in meta
    assert "\u00b7 \u00b7" not in meta and "control\u00b7" not in meta


@pytest.mark.parametrize("kind,heading,note", [
    ("prompt-leak", "System-Prompt Extraction", "All prompt-leak trials errored"),
    ("trust-escalation", "Trust Escalation", "All trust-escalation trials errored"),
    ("unsafe-output", "Unsafe Output Handling", "All unsafe-output trials errored"),
    ("dead-end", "Dead-End Scope-Holding", "All unsolvable dead-end trials errored"),
])
def test_all_errored_kind_is_disclosed_in_both_renderers(kind, heading, note):
    """A kind whose every trial errored must be SURFACED, not dropped. The HTML renderer gated three
    of these sections on the error-filtered list, so an unreachable target produced a shareable
    report with the section silently missing while the header above it still counted the kind."""
    rows = [_row(objective_id="o1", objective_kind="forbidden", verdict="refused"),
            _row(objective_id=f"o-{kind}", objective_kind=kind, verdict="error")]
    md, html = build_report(rows), build_html_report(rows)
    assert heading in md and heading in html, kind
    assert note in md and note in html, kind


@pytest.mark.parametrize("kind,note", [
    ("prompt-leak", "All prompt-leak trials errored"),
    ("trust-escalation", "All trust-escalation trials errored"),
    ("unsafe-output", "All unsafe-output trials errored"),
])
def test_errored_note_is_absent_when_trials_succeeded(kind, note):
    """The disclosure must not appear over real numbers. Gating the note on the raw kind list alone
    prints 'All <kind> trials errored' directly above that kind's results — a self-contradicting
    report that no positive assertion can catch."""
    rows = [_row(objective_id="o1", objective_kind="forbidden", verdict="refused"),
            _row(objective_id=f"o-{kind}", objective_kind=kind, verdict="refused")]
    md, html = build_report(rows), build_html_report(rows)
    assert note not in md, kind
    assert note not in html, kind
