"""MITRE ATLAS tags: validated loudly at load, kept out of the library fingerprints, and the README's
framework table is exactly what the library carries."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from iago import config
from iago.artifacts import library_fingerprint, scenario_fingerprint
from iago.attacks import load_library
from iago.campaign import SURFACES
from iago.config import VALID_ATLAS, tag_id, validate_atlas
from iago.frameworks import COVERAGE_END, COVERAGE_START, render_coverage_table
from iago.yaml_records import load_records

ROOT = Path(__file__).resolve().parent.parent

# Each registered agent surface's shipped scenario file, named explicitly.
SCENARIO_FILES = {
    "agent": config.AGENT_SCENARIOS_FILE, "toolabuse": config.TOOLABUSE_SCENARIOS_FILE,
    "privilege": config.PRIVILEGE_SCENARIOS_FILE, "disclosure": config.DISCLOSURE_SCENARIOS_FILE,
    "misinfo": config.MISINFO_SCENARIOS_FILE, "memory": config.MEMORY_SCENARIOS_FILE,
    "rag": config.RAG_SCENARIOS_FILE, "a2a": config.A2A_SCENARIOS_FILE,
}


def test_validate_atlas_accepts_a_known_id_with_or_without_a_label():
    assert validate_atlas("AML.T0054: LLM Jailbreak", where="x") == "AML.T0054: LLM Jailbreak"
    assert validate_atlas("AML.T0051.000", where="x") == "AML.T0051.000"
    assert validate_atlas(None, where="x") is None
    assert validate_atlas("  ", where="x") is None
    assert validate_atlas("AML.T0054 LLM Jailbreak", where="x")  # a space instead of a colon
    assert tag_id("AML.T0054 LLM Jailbreak") == tag_id("AML.T0054: LLM Jailbreak") == "AML.T0054"


@pytest.mark.parametrize("bad", ["AML.T9999: invented", "T0054: LLM Jailbreak", "AML.T0051: parent only",
                                 ": AML.T0054", "aml.t0054"])
def test_validate_atlas_rejects_an_unlisted_id_naming_where(bad):
    with pytest.raises(ValueError, match=r"somefile rec-1: invalid atlas tag"):
        validate_atlas(bad, where="somefile rec-1")


def test_the_atlas_list_holds_only_atlas_technique_ids():
    assert VALID_ATLAS and all(k.startswith("AML.T") for k in VALID_ATLAS)


def test_technique_loader_rejects_bad_atlas(tmp_path: Path):
    (tmp_path / "bad_atlas.yaml").write_text(textwrap.dedent("""\
        - id: bad-001
          name: Bad ATLAS tag
          category: role-play
          description: x
          atlas: "AML.T0000: nope"
          template: "do {objective}"
        """))
    with pytest.raises(ValueError, match="bad_atlas.yaml bad-001: invalid atlas tag"):
        load_library(tmp_path)


def test_every_registered_surface_has_a_named_scenario_file():
    assert set(SCENARIO_FILES) == set(SURFACES)


@pytest.mark.parametrize("key", sorted(SURFACES))
def test_every_scenario_loader_rejects_bad_atlas(key, tmp_path):
    """Each loader validates on its own code path; one forgetting would load anything."""
    import yaml
    src = SCENARIO_FILES[key]
    recs = load_records(src)
    next(r for r in recs if r.get("atlas"))["atlas"] = "AML.T0000: nope"
    bad = tmp_path / src.name
    bad.write_text(yaml.safe_dump(recs, sort_keys=False))
    with pytest.raises(ValueError, match="invalid atlas tag"):
        SURFACES[key].entry().load_scenarios(bad)


def test_provenance_forging_carries_asi09_and_an_atlas_tag():
    techs = [t for t in load_library() if t.category == "provenance-forging"]
    assert techs
    assert all(t.asi.startswith("ASI09") and t.atlas.startswith("AML.T0080.001") for t in techs)


def test_atlas_stays_out_of_the_technique_fingerprint():
    """A label edit must not make two runs of the same attack text look like different libraries."""
    from dataclasses import replace
    lib = load_library()
    assert any(t.atlas for t in lib)
    before = library_fingerprint(lib)
    assert library_fingerprint([replace(t, atlas="AML.T0054: LLM Jailbreak") for t in lib]) == before
    assert library_fingerprint([replace(lib[0], template=lib[0].template + " ")] + lib[1:]) != before


def test_atlas_stays_out_of_the_scenario_fingerprint():
    from dataclasses import replace
    scens = SURFACES["agent"].scenarios()
    assert scenario_fingerprint(scens) == scenario_fingerprint([replace(s, atlas=None) for s in scens])
    assert scenario_fingerprint(scens) != scenario_fingerprint([replace(scens[0], task=scens[0].task + " ")]
                                                               + scens[1:])


def test_the_replay_fixture_library_hash_still_matches():
    """The shipped fixture was recorded before ATLAS tags existed; its hash still describes the library."""
    header = json.loads((ROOT / "samples/replay/prompt-extraction.jsonl").read_text().split("\n", 1)[0])
    lib = [t for t in load_library() if t.category == "prompt-extraction"]
    assert all(t.atlas for t in lib)
    assert library_fingerprint(lib) == header["technique_library_sha256"]


def test_readme_framework_table_is_what_the_library_carries():
    readme = (ROOT / "README.md").read_text()
    assert readme.count(COVERAGE_START) == 1 and readme.count(COVERAGE_END) == 1
    shipped = readme[readme.index(COVERAGE_START):readme.index(COVERAGE_END) + len(COVERAGE_END)]
    assert shipped == render_coverage_table(), \
        "README framework table is stale: paste `uv run python -m iago.frameworks` between the markers"


def test_the_table_names_every_surface_and_every_atlas_id_in_use():
    table = render_coverage_table()
    used = {tag_id(t.atlas) for t in load_library() if t.atlas}
    for spec in SURFACES.values():
        assert f"`{spec.command}` scenarios" in table
        used |= {tag_id(s.atlas) for s in spec.scenarios() if s.atlas}
    for atlas_id in used:
        assert f"`{atlas_id}` {VALID_ATLAS[atlas_id]}" in table
