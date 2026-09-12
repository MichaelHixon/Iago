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

# Claim phrases that must carry the same figures in both copies. Comparing a flat positional
# tuple was not enough: swapping which figure attached to "fabricated a completion" and which to
# "used a machine stop channel" produced byte-identical tuples, so an inverted finding shipped
# green — and the ordering exemption that was supposed to cover it pointed the wrong way.
_CLAIM_PHRASES = (
    "fabricated a completion",
    "machine stop channel",
    "leaked the planted canary",
    "flipped a refusal to compliance",
    "emitted a live injectable construct",
    "confirmed bypasses",
)

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
    """One run exercising every shared surface AND every verdict.

    Row-shape diversity is load-bearing, not decoration. With nine clean rows every scorecard row
    read VULNERABLE, so hardcoding the verdict into the HTML pill survived the whole suite; and
    with no needs-review row, a diverged `leak_valid` filter rendered 2/3 (67%) in one copy and
    2/4 (50%) in the other while both parsers still agreed. One row of each shape turns both
    survivors into kills with no new assertion logic."""
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
        # an objective that HELD, so the verdict column is not a constant
        _row(objective_id="o-clean", trial=0, verdict="refused"),
        _row(objective_id="o-clean", trial=1, verdict="refused"),
        # a needs-review row: without one, the valid-row filters cannot be seen to diverge
        _row(objective_id="o-leak", objective_kind="prompt-leak", technique_id="t2", trial=1,
             verdict="needs-review"),
        # an errored row, excluded from every rate
        _row(objective_id="o-forb", trial=3, verdict="error"),
        # a decode-failed row: reached the model but never exercised the guardrail
        _row(objective_id="o-forb", trial=4, verdict="refused", gated=True, decoded=False),
    ]


def test_the_extractor_reads_what_a_human_reads():
    """A differential is worthless if the parser silently finds nothing — every comparison becomes
    () == () and every mutation passes. Pin the extractor against a hand-read of the fixture."""
    md_doc = build_report(_fixture())
    summary = _exec_summary(md_doc)
    # o-forb contributes 3 valid trials (its errored and decode-failed rows are excluded) and
    # o-clean 2, so 1 confirmed bypass out of 5 valid forbidden trials, at 20%
    assert _numbers_after(summary, "Harmful content")[:3] == ("1", "5", "20%")
    # the leak objective has one bypass and one needs-review row, both valid
    assert _numbers_after(summary, "System-prompt extraction (LLM07)")[:2] == ("1", "2")
    assert _numbers_after(summary, "Nonexistent label") is None

    rows = _scorecard_rows(md_doc, is_html=False)
    assert len(rows) == 6, rows       # o-forb, o-clean, leak, trust, unsafe, dead-end
    forb = next(r for r in rows if r[1] == "o-forb")
    assert forb[2] == "VULNERABLE" and "1/3" in forb[3], forb
    # the verdict column is not a constant, which is what makes comparing it meaningful
    assert next(r for r in rows if r[1] == "o-clean")[2] == "INCONCLUSIVE", rows

    assert _header_count(md_doc, "scored trials") == "11", _header_count(md_doc, "scored trials")


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


def _claim_figures(text: str) -> dict:
    """{claim phrase: figures in the clause that makes it}, for every phrase present.

    Positional comparison cannot see a claim INVERSION — swap which figure attaches to "fabricated
    a completion" and which to "machine stop channel" and the flat tuple is unchanged, so the
    shared copy can report the exact inverse of the finding and stay green. Clauses are split on
    `;` because that is how both renderers join two claims in one sentence."""
    out = {}
    for clause in re.split(r"[;.] ", text):
        for phrase in _CLAIM_PHRASES:
            if phrase in clause:
                out.setdefault(phrase, []).append(tuple(re.findall(_FIGURE, clause)))
    return out


def test_each_claim_carries_the_same_figures_in_both_renderers():
    """The inversion test. Swapping the dead-end fabrication and stop figures in one renderer left
    all 945 tests green while the two copies reported opposite findings about the same run."""
    rows = _fixture()
    md = _claim_figures(_exec_summary(build_report(rows)))
    html = _claim_figures(_exec_summary(build_html_report(rows)))
    assert md, "no claim phrases found in markdown — the parser found nothing"
    assert set(md) == set(html), (sorted(md), sorted(html))
    for phrase in md:
        assert md[phrase] == html[phrase], (phrase, md[phrase], html[phrase])


@pytest.mark.parametrize("renderer", [build_report, build_html_report], ids=["md", "html"])
def test_every_figure_in_a_report_is_internally_consistent(renderer):
    """Not a differential — a differential structurally CANNOT catch a shared helper. Swapping the
    bounds inside `ci_str` reverses every confidence interval in BOTH copies at once, so the two
    still agree and the suite stays green. These assertions read one document on its own terms."""
    doc = _plain(renderer(_fixture()))

    for a, b in re.findall(r"(\d+) ?/ ?(\d+)", doc):
        assert int(a) <= int(b), f"{a}/{b} claims more events than trials"

    for lo, hi in re.findall(r"(\d+)%[–-](\d+)%", doc):
        assert int(lo) <= int(hi), f"confidence interval {lo}%–{hi}% is reversed"

    for conf in re.findall(r"\bconfidence[^0-9]{0,20}(\d+\.\d+)", doc):
        assert 0.0 <= float(conf) <= 1.0, f"confidence {conf} outside [0, 1]"


# --- ISC-55: each section's own headline, and the surfaces one renderer lacks -----------------

# Headings markdown emits that HTML does not, each with the reason it is absent. This is the
# allowlist ISC-54's anti-criterion called for and did not have: `_SHARED_LABELS` is an inclusion
# list, so a one-copy surface was invisible to it rather than named.
_MARKDOWN_ONLY_HEADINGS = {
    # (Judge calibration and Decode-gating are markdown H3s, not sections — HTML renders the same
    # figures inline in the executive summary, and test_isc53.py pins both.)
    "Bypass Rate by Category": "no HTML equivalent — the shared copy has no category breakdown",
    "Bypass Rate by Technique": "no HTML equivalent — the shared copy has no technique breakdown",
    "Evidence — Strongest Harmful-Content Bypasses":
        "HTML carries its excerpts inside each finding section instead of one evidence section",
}


def _headings(doc: str, is_html: bool) -> list[str]:
    if is_html:
        return re.findall(r"<h2>(.*?)</h2>", doc)
    # H2 only: H3s are sub-parts, and the numbered evidence items are H3s per item
    return [h.strip() for h in re.findall(r"^## (.+)$", doc, re.M)]


def _section_headline(doc: str, heading: str, is_html: bool) -> tuple[str, ...] | None:
    """(numerator, denominator, rate) from a section's headline claim, or None if absent.

    Both renderers open every finding section with the same `a / b … p%` shape, then diverge in
    wording ("exfiltrated the canary — 50%" vs "leaked (50%…)"), so the figures are compared and
    the prose is not. Scoped to the headline claim rather than the whole section on purpose:
    markdown legitimately carries sub-tables HTML does not, and folding those in would force the
    copies together on a real structural difference — where the tempting fix is deleting the
    markdown table rather than adding it to HTML."""
    key = f"<h2>{heading}</h2>" if is_html else f"## {heading}"
    if key not in doc:
        return None
    text = _plain(doc.split(key, 1)[1][:5000])
    m = re.search(r"(\d+) / (\d+)[^.]{0,80}?(\d+)%", text)
    return m.groups() if m else None


def test_the_markdown_only_allowlist_matches_what_the_renderers_emit():
    """The allowlist is only honest if it is checked against reality. A section added to HTML later
    must not sit silently in the exemption list, and one added to markdown must not bypass the
    differential by being forgotten."""
    rows = _fixture()
    md_h = set(_headings(build_report(rows), is_html=False))
    html_h = set(_headings(build_html_report(rows), is_html=True))
    assert set(_MARKDOWN_ONLY_HEADINGS) == md_h - html_h, {
        "listed but now shared or gone": set(_MARKDOWN_ONLY_HEADINGS) - (md_h - html_h),
        "markdown-only but unlisted": (md_h - html_h) - set(_MARKDOWN_ONLY_HEADINGS),
    }
    assert not html_h - md_h, f"HTML-only headings are not expected: {html_h - md_h}"


@pytest.mark.parametrize("heading", [
    "System-Prompt Extraction (OWASP LLM07)",
    "Trust Escalation — Fabricated Provenance (OWASP LLM01)",
    "Unsafe Output Handling (OWASP LLM05)",
    "Dead-End Scope-Holding (fabricated task completion)",
])
def test_each_section_headline_agrees_with_the_other_renderer(heading):
    """Every per-section headline restates its executive-summary figure, and all four were
    unpinned. A divergence does not just print a wrong number — it makes the shared page
    contradict itself, "1 / 1 trials leaked" at the top and "5 / 10" four sections down."""
    rows = _fixture()
    md = _section_headline(build_report(rows), heading, is_html=False)
    html = _section_headline(build_html_report(rows), heading, is_html=True)
    assert md, f"no figures parsed from the markdown {heading} headline"
    assert md == html, (heading, md, html)


def test_technique_caveats_use_the_same_row_filter_in_both_renderers():
    """The HTML caveats were gated on `valid` — forbidden, non-errored, non-decode-failed — while
    markdown gates on every row. So the shared copy dropped both caveats on an errored
    template-injection trial and on a many-shot trial against a leak objective, which is exactly
    the divergence the block's own comment claims to have closed."""
    rows = [_row(objective_id="o1", verdict="refused"),
            _row(objective_id="o1", technique_id="ti-001", category="template-injection",
                 verdict="error"),
            _row(objective_id="ol", objective_kind="prompt-leak", technique_id="ms-001",
                 category="many-shot", verdict="bypassed", shots=64)]
    for doc, name in zip((build_report(rows), build_html_report(rows)), ("md", "html")):
        assert "control-token injection" in doc, name
        assert "repetition-driven" in doc, name


def test_all_three_calibration_figures_are_pinned():
    """Two of the three numbers in the judge-calibration sentence were covered; the needs-review
    count in the middle was not, so it could read 1/2 where the truth is 0/2."""
    rows = [_row(objective_id="o1", verdict="refused"),
            _row(objective_id="oc", objective_kind="control", trial=0, verdict="refused"),
            _row(objective_id="oc", objective_kind="control", trial=1, verdict="needs-review")]
    for doc, name in zip((build_report(rows), build_html_report(rows)), ("md", "html")):
        plain = _plain(doc)
        i = plain.index("benign control was confirmed")
        sentence = plain[i:plain.index("harmless", i)]
        assert "bypassed in 0/2" in sentence, (name, sentence)
        assert "needs-review in 1/2" in sentence, (name, sentence)


# --- ISC-55: the transcripts, which had no cross-renderer test at all -------------------------

def _log_meta(doc: str, is_html: bool) -> list[dict]:
    """The per-trial metadata line from either transcript, as {field: value} per trial."""
    if is_html:
        raw = re.findall(r"<div class=sub>(kind .*?)</div>", doc)
    else:
        raw = [l[2:] for l in doc.splitlines() if l.startswith("- kind `")]
    out = []
    for line in raw:
        fields = {}
        for part in _plain(line).split("·"):
            bits = part.strip().split(" ", 1)
            if len(bits) == 2:
                fields[bits[0]] = bits[1].strip()
        out.append(fields)
    return out


def _log_fixture():
    """Two trials that differ ONLY in temperature and decode status — the pair that rendered
    byte-identical in the HTML transcript."""
    return [_row(objective_id="o1", trial=0, temperature=0.0, verdict="refused",
                 gated=True, decoded=False, confidence=0.4),
            _row(objective_id="o1", trial=1, temperature=1.2, verdict="bypassed",
                 gated=True, decoded=True, confidence=0.9)]


def test_the_two_transcripts_carry_the_same_per_trial_metadata():
    """`build_log` and `build_html_log` had no cross-renderer test of any kind, and the HTML copy
    omitted `temp` and `decoded` — so two trials differing only in sampling temperature and decode
    status rendered identically in the copy a reviewer checks the report against."""
    from iago.report import build_html_log, build_log

    rows = _log_fixture()
    md, html = _log_meta(build_log(rows), is_html=False), _log_meta(build_html_log(rows), is_html=True)
    assert len(md) == len(rows) == len(html), (len(md), len(html))
    assert md == html, (md, html)
    # and the two trials are actually distinguishable in the HTML copy
    assert html[0] != html[1], html


@pytest.mark.parametrize("renderer,is_html", [("build_log", False), ("build_html_log", True)],
                         ids=["md", "html"])
def test_transcript_confidence_stays_within_range(renderer, is_html):
    """Doubling the rendered confidence printed 1.60 and survived the whole suite."""
    import iago.report as _r

    for meta in _log_meta(getattr(_r, renderer)(_log_fixture()), is_html):
        c = float(meta["confidence"])
        assert 0.0 <= c <= 1.0, meta


def test_both_transcripts_report_the_same_trial_count():
    """A transcript claiming ten trials where the other claims nine."""
    from iago.report import build_html_log, build_log

    rows = _log_fixture()
    # each renderer phrases it its own way: "Total trials: 2" vs "2 trials"
    md_n = _header_count(build_log(rows), "total trials")
    html_n = _header_count(build_html_log(rows), "trials")
    assert md_n and html_n, (md_n, html_n)
    assert md_n == html_n == str(len(rows)), (md_n, html_n)


# --- ISC-55: the four breakdown tables, the entire drill-down of the shared copy ---------------

# Tables one renderer emits and the other does not, each with its reason. Same contract as
# _MARKDOWN_ONLY_HEADINGS: named, not silently skipped.
_MARKDOWN_ONLY_TABLES = {
    "Category": "no HTML equivalent — the shared copy has no category breakdown",
    "Rank": "no HTML equivalent — the shared copy has no technique breakdown",
    "Gated technique": "no HTML equivalent — the shared copy has no per-technique decode table",
}


def _cell_figures(text: str) -> frozenset:
    """Figures in a table row, with fractions expanded to their parts.

    The renderers group the same values differently by design — markdown gives `Leaked | Trials`
    two columns where HTML writes `1/2` in one — so comparing raw figure strings would fail on a
    layout choice. Expanding means a wrong VALUE still fails (every measured survivor was a value
    change: a numerator +1, a denominator +1, a halved rate) while a pure grouping difference does
    not. A transposition inside an HTML fraction is caught separately, by the `a <= b`
    internal-consistency check."""
    out = set()
    for fig in re.findall(_FIGURE, text):
        if "/" in fig:
            out.update(fig.split("/"))
        else:
            out.add(fig)
    return frozenset(out)


def _tables(doc: str, is_html: bool) -> dict:
    """{first column header: {row label: frozenset(figures)}} for every table in the document.

    Keyed on the row LABEL rather than position, so a reordered table is not a false failure —
    row order legitimately differed between renderers until the sort tiebreaks were matched, and
    the figures are what this asserts."""
    out = {}
    if is_html:
        for tbl in re.findall(r"<table.*?</table>", doc, re.S):
            heads = [_plain(c).strip() for c in re.findall(r"<th[^>]*>(.*?)</th>", tbl, re.S)]
            if not heads:
                continue
            rows = {}
            for tr in re.findall(r"<tr>(.*?)</tr>", tbl, re.S):
                cells = [_plain(c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                if cells:
                    rows[cells[0]] = _cell_figures(" ".join(cells[1:]))
            if rows:
                out[heads[0]] = rows
        return out
    block, header = [], None
    for line in doc.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if len(block) >= 3:
            header = [c.strip() for c in block[0].strip("|").split("|")]
            rows = {}
            for line2 in block[2:]:
                cells = [_plain(c).strip() for c in line2.strip("|").split("|")]
                if cells:
                    rows[cells[0]] = _cell_figures(" ".join(cells[1:]))
            if rows:
                out[header[0]] = rows
        block = []
    return out


def _table_fixture():
    """Two techniques per finding kind, so every breakdown table renders more than one row."""
    rows = list(_fixture())
    for i, (kind, tid) in enumerate([("prompt-leak", "lk"), ("trust-escalation", "tr"),
                                     ("unsafe-output", "us"), ("dead-end", "de")]):
        for j in range(2):
            rows.append(_row(objective_id=f"t-{kind}-{j}", objective_kind=kind,
                             technique_id=f"{tid}-{j}", trial=j,
                             verdict="bypassed" if j == 0 else "refused",
                             **({"sink": "html" if j == 0 else "shell"} if kind == "unsafe-output" else {}),
                             **({"deadend_signal": "result-channel", "deadend_control": False}
                                if kind == "dead-end" else {})))
    return rows


def test_the_breakdown_table_allowlist_matches_what_the_renderers_emit():
    """Same contract as the heading allowlist: a table added to HTML later must not sit silently
    in the exemption list."""
    rows = _table_fixture()
    md_t = set(_tables(build_report(rows), is_html=False))
    html_t = set(_tables(build_html_report(rows), is_html=True))
    assert set(_MARKDOWN_ONLY_TABLES) == md_t - html_t, {
        "listed but now shared or gone": set(_MARKDOWN_ONLY_TABLES) - (md_t - html_t),
        "markdown-only but unlisted": (md_t - html_t) - set(_MARKDOWN_ONLY_TABLES),
    }


def test_every_shared_breakdown_table_agrees_cell_for_cell():
    """Seven separate cell mutations survived the suite, including denominators — a "2/1" cell
    shipped green. These four tables are the whole drill-down of the shared copy, because HTML has
    no category or technique breakdown at all."""
    rows = _table_fixture()
    md_t = _tables(build_report(rows), is_html=False)
    html_t = _tables(build_html_report(rows), is_html=True)
    shared = set(md_t) & set(html_t)
    assert shared, (sorted(md_t), sorted(html_t))
    for key in sorted(shared):
        assert md_t[key] == html_t[key], (key, md_t[key], html_t[key])


# --- ISC-55: the recommendation list, which one renderer could silently shorten ----------------

def _recommendations(doc: str, is_html: bool) -> list[str]:
    """The leading phrase of each hardening recommendation, in order.

    The leading phrase rather than the whole text: both renderers open each item with the same
    bolded directive and then diverge in the supporting sentence, so this pins WHICH advice is
    given and how much of it, without forcing the prose together."""
    if is_html:
        body = doc[doc.index("Hardening Recommendations"):]
        items = re.findall(r"<li>(.*?)</li>", body, re.S)
    else:
        body = doc[doc.index("## Hardening Recommendations"):]
        items = re.findall(r"^\d+\. (.+)$", body, re.M)
    out = []
    for item in items:
        text = _plain(item).strip()
        out.append(re.split(r"[.:]", text, maxsplit=1)[0].strip())
    return out


def test_the_same_recommendations_reach_both_renderers():
    """Truncating the list in one copy — HTML dropping its last recommendation, or `_hardening_recs`
    taking [:1] instead of [:3] — survived the whole suite. The recommendations are the actionable
    half of the report and the shared copy is the one that gets acted on."""
    rows = _table_fixture()
    md = _recommendations(build_report(rows), is_html=False)
    html = _recommendations(build_html_report(rows), is_html=True)
    assert md, "no recommendations parsed from markdown"
    assert md == html, (md, html)
