"""The hardening recommendations name a concrete defense for every category that leaked, in both
renderers, and promise no work that has not been built."""

import html as htmllib
import re

import pytest

from iago.attacks import load_library
from iago.report import CATEGORY_DEFENSES, _GENERIC_DEFENSE, build_html_report, build_report


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
    text = htmllib.unescape(re.sub(r"<[^>]+>", "", doc))
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def _first_words(defense: str) -> str:
    """A distinctive prefix of the defense, normalized the way _plain normalizes the report."""
    return _plain(defense)[:50]


def _harmful_categories() -> list[str]:
    """Categories with a technique that can fire at a harmful-content objective: the only ones a
    leaking-category recommendation can ever name."""
    return sorted({t.category for t in load_library()
                   if t.applies_to is None or "forbidden" in t.applies_to})


def test_every_harmful_content_category_has_its_own_defense_and_no_entry_is_dead():
    """A new category without a defense would silently get the generic line; an entry for a category
    that can never leak harmful content is text no report can show."""
    harmful = _harmful_categories()
    assert harmful, "no harmful-content categories loaded"
    assert sorted(CATEGORY_DEFENSES) == harmful
    assert len(set(CATEGORY_DEFENSES.values())) == len(CATEGORY_DEFENSES), "two categories share a defense"


@pytest.mark.parametrize("render", [build_report, build_html_report], ids=["markdown", "html"])
@pytest.mark.parametrize("category", sorted(CATEGORY_DEFENSES))
def test_a_leaking_category_gets_its_defense_in_both_renderers(render, category):
    text = _plain(render([_row(category=category, technique_id=f"{category}-1")]))
    assert f"Defend {category}:" in text
    assert _first_words(CATEGORY_DEFENSES[category]) in text


@pytest.mark.parametrize("render", [build_report, build_html_report], ids=["markdown", "html"])
def test_every_leaking_category_is_defended_not_only_the_top_three(render):
    cats = ["role-play", "format-shift", "many-shot", "multi-turn", "encoding-obfuscation"]
    rows = [_row(category=c, technique_id=f"{c}-1", objective_id=f"o-{c}") for c in cats]
    rows.append(_row(category="direct-ask", technique_id="da-1", verdict="refused"))
    text = _plain(render(rows))
    for c in cats:
        assert f"Defend {c}:" in text, c
    assert "Defend direct-ask:" not in text, "a category with no bypass was given a defense"


@pytest.mark.parametrize("render", [build_report, build_html_report], ids=["markdown", "html"])
def test_defenses_follow_the_bypass_rate_highest_first(render):
    rows = [_row(category="role-play", technique_id="rp-1", objective_id="o1"),
            _row(category="many-shot", technique_id="ms-1", objective_id="o2"),
            _row(category="many-shot", technique_id="ms-1", objective_id="o3", verdict="refused")]
    text = _plain(render(rows))  # role-play 1/1, many-shot 1/2; alphabetical order would flip them
    assert text.index("Defend role-play:") < text.index("Defend many-shot:")


def test_the_html_copy_renders_the_defense_heading_as_markup_not_raw_markdown():
    html = build_html_report([_row(category="role-play", technique_id="rp-1")])
    assert "<strong>Defend <code>role-play</code>:</strong>" in html
    assert "**Defend" not in html


@pytest.mark.parametrize("render", [build_report, build_html_report], ids=["markdown", "html"])
def test_an_unmapped_category_gets_the_generic_defense_not_silence(render):
    text = _plain(render([_row(category="custom-cat", technique_id="cc-1")]))
    assert "Defend custom-cat:" in text
    assert _first_words(_GENERIC_DEFENSE) in text


@pytest.mark.parametrize("render", [build_report, build_html_report], ids=["markdown", "html"])
def test_no_recommendation_promises_planned_work(render):
    text = _plain(render([_row()]))
    assert "(planned)" not in text
    assert "Upgrade the judge" not in text
