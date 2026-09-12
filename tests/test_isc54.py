"""ISC-54 — every number the two renderers share is pinned to be EQUAL, by one invariant.

ISC-53 closed *which* disclosures each renderer makes. This closes *what numbers* they print.
Measured before it existed: `+1` on the HTML executive summary's harmful-content numerator, and on
its LLM07 denominator, each left all 936 tests green — while the same mutation in markdown failed.

The invariant is a differential: it compares the two rendered documents to each other and never
re-derives an expected value from the production code, so it cannot decay into a tautology. It does
not require the copies to be byte-identical — they legitimately differ in prose, ordering and cell
layout — only to agree on every figure they both claim.
"""

import html as htmllib
import re
from collections import Counter

import pytest

from iago.report import build_html_report, build_report

# Executive-summary labels both renderers emit for the same finding. A surface present in only one
# copy surfaces as a missing label rather than as a silently skipped comparison.
_SHARED_LABELS = (
    "Harmful content",
    "System-prompt extraction (LLM07)",
    "Trust escalation",
    "Unsafe output handling (LLM05)",
    "Dead-end scope-holding",
)

# Labels whose sentences present the SAME figures in a different order by design. Named here
# rather than silently tolerated: every other label is compared ordered, so a swapped numerator
# and denominator fails.
_ORDER_MAY_DIFFER = {
    "Dead-end scope-holding": "markdown leads with the stop rate, HTML with the fabrication rate",
}

# "1/3", "33%", "21%–100%", "6%-79%"
_FIGURE = r"\d+(?:\.\d+)?%?(?:/\d+)?(?:[–-]\d+(?:\.\d+)?%?)?"


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


def _plain(doc: str) -> str:
    """One normal form for both renderers: no tags, no entities, no emphasis markers, one space.

    Deliberately lossy about PRESENTATION and lossless about NUMBERS — the invariant is only ever
    about the figures the two copies share."""
    text = re.sub(r"<[^>]+>", " ", doc)
    text = htmllib.unescape(text)
    text = text.replace("**", "").replace("`", "").replace("*", "")
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text)


def _exec_summary(doc: str) -> str:
    """The executive-summary section only, normalized.

    Sliced BEFORE normalizing, because the same labels also head scorecard rows — searching the
    whole document found those instead, and their cell order differs by design."""
    if "<h2>Executive Summary</h2>" in doc:
        body = doc.split("<h2>Executive Summary</h2>", 1)[1].split("<h2>", 1)[0]
    else:
        body = doc.split("## Executive Summary", 1)[1]
        body = re.split(r"\n## |\n<details>", body, maxsplit=1)[0]
    return _plain(body)


def _numbers_after(text: str, label: str) -> tuple[str, ...] | None:
    """Every figure in the sentence introduced by `label`, or None when the label is absent.

    Figures stay strings: "50%" and "9%–91%" carry their units, and a renderer printing 50 where
    the other printed 50% is a real divergence, not a formatting nicety."""
    i = text.find(label)
    if i == -1:
        return None
    sentence = text[i + len(label):]                # after the label: "LLM07" is not a figure
    end = sentence.find(". ")
    sentence = sentence[:end] if end != -1 else sentence[:400]
    # framework identifiers are names, not measurements — markdown's LLM07 line carries a trailing
    # "detail in the LLM07 section below" that HTML has no equivalent for
    sentence = re.sub(r"\b(?:LLM|ASI)\d+\b", "", sentence)
    return tuple(re.findall(_FIGURE, sentence))


def _header_count(doc: str, noun: str) -> str | None:
    """A header count, however each renderer phrases it: "Techniques: 4" or "4 techniques"."""
    text = _plain(doc)
    m = (re.search(rf"{noun}:\s*(\d+)", text, re.IGNORECASE)
         or re.search(rf"(\d+)\s+{noun}", text, re.IGNORECASE))
    return m.group(1) if m else None


def _row_key(cells: list[str]) -> tuple:
    """(surface, objective, verdict, {figures}) for one scorecard row.

    Identity is ordered; figures are a set. The renderers lay the numeric cells out differently on
    purpose — markdown prints `Confirmed | Rate | CI` as three cells, HTML prints `100% (1/1)` in
    one — so requiring positional equality would fail on a formatting choice rather than a defect.
    """
    surface, objective = cells[0].strip(), cells[1].strip()
    verdict = re.sub(r"[^A-Z]", "", cells[2])          # markdown prefixes an emoji
    figures = frozenset(re.findall(_FIGURE, " ".join(cells[3:])))
    return (surface, objective, verdict, figures)


def _scorecard_rows(doc: str, is_html: bool) -> list[tuple]:
    """One `_row_key` per scored objective, from either copy."""
    if is_html:
        body = doc[doc.index("<h2>Scorecard"):]
        body = body.split("</table>", 1)[0]
        out = []
        for r in re.findall(r"<tr>(.*?)</tr>", body, re.S):
            cells = [_plain(c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
            if len(cells) >= 4 and cells[0] != "Surface":
                out.append(_row_key(cells))
        return out
    body = doc[doc.index("## Scorecard"):]
    body = body.split(chr(10) + '## ', 1)[0]      # this section only, not every later table
    out = []
    for line in body.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [_plain(c).strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4 and cells[0] != "Surface":
            out.append(_row_key(cells))
    return out


def _fixture():
    """One run exercising every shared surface: a forbidden objective with a bypass and two holds,
    a control, a leak, a trust escalation, an unsafe-output emission, and a dead-end unsolvable
    alongside its solvable control."""
    return [
        _row(objective_id="o-forb", trial=0, verdict="bypassed"),
        _row(objective_id="o-forb", trial=1, verdict="refused"),
        _row(objective_id="o-forb", trial=2, verdict="refused"),
        _row(objective_id="o-ctl", objective_kind="control", verdict="refused"),
        _row(objective_id="o-leak", objective_kind="prompt-leak", technique_id="t2",
             verdict="bypassed"),
        _row(objective_id="o-trust", objective_kind="trust-escalation", technique_id="t3",
             verdict="bypassed", placebo_signal="refused"),
        _row(objective_id="o-unsafe", objective_kind="unsafe-output", technique_id="t4",
             verdict="bypassed", sink="html"),
        _row(objective_id="o-de", objective_kind="dead-end", technique_id="t5",
             verdict="bypassed", deadend_signal="result-channel", deadend_control=False),
        _row(objective_id="o-de-c", objective_kind="dead-end", technique_id="t5",
             verdict="refused", deadend_signal="control-correct", deadend_control=True),
    ]


def test_the_extractor_reads_what_a_human_reads():
    """A differential is worthless if the parser silently finds nothing — every comparison becomes
    () == () and every mutation passes. Pin the extractor against a hand-read of the fixture."""
    md_doc = build_report(_fixture())
    summary = _exec_summary(md_doc)
    # 1 of 3 forbidden trials was a confirmed bypass, at 33%
    assert _numbers_after(summary, "Harmful content")[:3] == ("1", "3", "33%")
    assert _numbers_after(summary, "System-prompt extraction (LLM07)")[:2] == ("1", "1")
    assert _numbers_after(summary, "Nonexistent label") is None

    rows = _scorecard_rows(md_doc, is_html=False)
    assert len(rows) == 5, rows            # forbidden, leak, trust, unsafe, dead-end
    forb = next(r for r in rows if r[1] == "o-forb")
    assert forb[2] == "VULNERABLE" and "1/3" in forb[3], forb

    # 9 fixture rows, one of them a control, which is not scored
    assert _header_count(md_doc, "scored trials") == "8", _header_count(md_doc, "scored trials")


@pytest.mark.parametrize("label", _SHARED_LABELS)
def test_executive_summary_numbers_agree_across_renderers(label):
    """Mutating the HTML harmful-content numerator, or its LLM07 denominator, left the whole suite
    green while markdown's equivalents were covered. This is that asymmetry closed."""
    rows = _fixture()
    md = _exec_summary(build_report(rows))
    html = _exec_summary(build_html_report(rows))
    md_nums, html_nums = _numbers_after(md, label), _numbers_after(html, label)
    assert md_nums is not None, f"{label} missing from markdown"
    assert html_nums is not None, f"{label} missing from HTML"
    # every figure, with multiplicity: a changed, dropped or invented number fails here whatever
    # the sentence order
    assert Counter(md_nums) == Counter(html_nums), (label, md_nums, html_nums)
    if label not in _ORDER_MAY_DIFFER:
        assert md_nums == html_nums, (label, md_nums, html_nums)


def test_scorecard_rows_agree_across_renderers():
    """The other surface carrying a verdict per objective. Dropping a row, or halving a rate, in
    the HTML copy left the suite green."""
    rows = _fixture()
    md_rows = _scorecard_rows(build_report(rows), is_html=False)
    html_rows = _scorecard_rows(build_html_report(rows), is_html=True)
    assert md_rows, "no scorecard rows parsed from markdown"
    assert sorted(md_rows) == sorted(html_rows), (md_rows, html_rows)


@pytest.mark.parametrize("noun", ["techniques", "scored trials"])
def test_header_counts_agree_across_renderers(noun):
    """The counts both copies print in the header, above everything else on the page."""
    rows = _fixture()
    md_n = _header_count(build_report(rows), noun)
    html_n = _header_count(build_html_report(rows), noun)
    assert md_n is not None and html_n is not None, (noun, md_n, html_n)
    assert md_n == html_n, (noun, md_n, html_n)
