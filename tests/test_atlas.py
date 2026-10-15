"""MITRE ATLAS tags: validated loudly at load, kept out of the library fingerprints, and the README's
framework table is exactly what the library carries."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from iago.artifacts import fingerprint_fields, scenario_fingerprint, sha256_text
from iago.attacks import load_library
from iago.config import ATLAS_RELEASE, VALID_ATLAS, validate_atlas
from iago.frameworks import COVERAGE_END, COVERAGE_START, SCENARIO_SURFACES, render_coverage_table

ROOT = Path(__file__).resolve().parent.parent


def test_validate_atlas_accepts_a_known_id_with_or_without_a_label():
    assert validate_atlas("AML.T0054: LLM Jailbreak", where="x") == "AML.T0054: LLM Jailbreak"
    assert validate_atlas("AML.T0051.000", where="x") == "AML.T0051.000"
    assert validate_atlas(None, where="x") is None
    assert validate_atlas("  ", where="x") is None


@pytest.mark.parametrize("bad", ["AML.T9999: invented", "T0054: LLM Jailbreak", "AML.T0051: parent only",
                                 ": AML.T0054", "aml.t0054"])
def test_validate_atlas_rejects_an_unlisted_id_naming_where(bad):
    with pytest.raises(ValueError, match=r"somefile rec-1: invalid atlas tag"):
        validate_atlas(bad, where="somefile rec-1")


def test_the_atlas_list_names_its_release():
    assert ATLAS_RELEASE == "2026.09"
    assert all(k.startswith("AML.T") for k in VALID_ATLAS)


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


@pytest.mark.parametrize("command,load", SCENARIO_SURFACES, ids=[c for c, _ in SCENARIO_SURFACES])
def test_every_scenario_loader_rejects_bad_atlas(command, load, tmp_path, monkeypatch):
    """Each of the eight loaders validates on its own code path; one forgetting would load anything."""
    real = load()
    assert real, command
    import yaml
    src = Path(load.__globals__[next(k for k in load.__globals__ if k.endswith("SCENARIOS_FILE"))])
    recs = yaml.safe_load(src.read_text())
    tagged = next(r for r in recs if r.get("atlas"))
    tagged["atlas"] = "AML.T0000: nope"
    bad = tmp_path / src.name
    bad.write_text(yaml.safe_dump(recs, sort_keys=False))
    with pytest.raises(ValueError, match="invalid atlas tag"):
        load(bad)


def test_provenance_forging_carries_asi09_and_an_atlas_tag():
    techs = [t for t in load_library() if t.category == "provenance-forging"]
    assert techs
    assert all(t.asi.startswith("ASI09") and t.atlas.startswith("AML.T0080.001") for t in techs)


def test_atlas_stays_out_of_the_technique_fingerprint():
    """A label edit must not make two runs of the same attack text look like different libraries."""
    lib = load_library()
    assert any(t.atlas for t in lib)
    before = sha256_text(json.dumps([fingerprint_fields(t) for t in lib], sort_keys=True, default=str))
    from dataclasses import replace
    relabeled = [replace(t, atlas="AML.T0054: LLM Jailbreak") for t in lib]
    after = sha256_text(json.dumps([fingerprint_fields(t) for t in relabeled], sort_keys=True, default=str))
    assert before == after
    changed = [replace(lib[0], template=lib[0].template + " ")] + lib[1:]
    assert sha256_text(json.dumps([fingerprint_fields(t) for t in changed], sort_keys=True, default=str)) != before


def test_atlas_stays_out_of_the_scenario_fingerprint():
    from dataclasses import replace
    _, load = SCENARIO_SURFACES[0]
    scens = load()
    assert scenario_fingerprint(scens) == scenario_fingerprint([replace(s, atlas=None) for s in scens])
    assert scenario_fingerprint(scens) != scenario_fingerprint([replace(scens[0], task=scens[0].task + " ")]
                                                               + scens[1:])


def test_the_replay_fixture_library_hash_still_matches():
    """The shipped fixture was recorded before ATLAS tags existed; its hash still describes the library."""
    header = json.loads((ROOT / "samples/replay/prompt-extraction.jsonl").read_text().split("\n", 1)[0])
    lib = [t for t in load_library() if t.category == "prompt-extraction"]
    assert all(t.atlas for t in lib)
    now = sha256_text(json.dumps([fingerprint_fields(t) for t in lib], sort_keys=True, default=str))
    assert now == header["technique_library_sha256"]


def test_readme_framework_table_is_what_the_library_carries():
    readme = (ROOT / "README.md").read_text()
    assert readme.count(COVERAGE_START) == 1 and readme.count(COVERAGE_END) == 1
    shipped = readme[readme.index(COVERAGE_START):readme.index(COVERAGE_END) + len(COVERAGE_END)]
    assert shipped == render_coverage_table(), \
        "README framework table is stale: paste `uv run python -m iago.frameworks` between the markers"


def test_the_table_names_every_surface_and_every_atlas_id_in_use():
    table = render_coverage_table()
    used = {t.atlas.split(":")[0] for t in load_library() if t.atlas}
    for command, load in SCENARIO_SURFACES:
        assert f"`{command}` scenarios" in table
        used |= {s.atlas.split(":")[0] for s in load() if s.atlas}
    for atlas_id in used:
        assert f"`{atlas_id}` {VALID_ATLAS[atlas_id]}" in table
