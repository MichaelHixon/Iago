"""ISC-50 — the determinism measurement reaches the reader, in every state.

ISC-49 made the probe able to fail. Nothing read the result: `build_report` took rows only, and
`compare` held the manifest but only looked at `judge_id`. So a host that had just measured itself
non-reproducible still produced a clean-looking pentest report and a clean-looking model delta.
Per ISC-40/47 these pin BEHAVIOR — the sentence must appear for the state that produced it, and
must NOT appear for the states that did not.
"""

import pytest

from iago.report import build_report, determinism_disclosure

MISMATCH = {"determinism": {"mismatch_detected": True, "options": {"seed": 1337},
                            "probes": [{"mismatch": False}, {"mismatch": True}]}}
CLEAN = {"determinism": {"mismatch_detected": False, "options": {"seed": 1337},
                         "probes": [{"mismatch": False}, {"mismatch": False}]}}
UNKNOWN = {"determinism": {"mismatch_detected": None, "options": {"seed": 1337},
                           "probes": [{"mismatch": None}, {"mismatch": False}]}}
SKIPPED = {"determinism": None}


def test_a_measured_mismatch_is_stated_and_names_the_probe_and_seed():
    text = determinism_disclosure(MISMATCH)
    assert "NOT bit-reproducible" in text
    assert "probe 2 of 2" in text and "seed 1337" in text
    # The consequence, not just the fact: a delta may be the host.
    assert "may be the host rather than the model" in text


def test_a_clean_result_is_never_stated_as_reproducible():
    text = determinism_disclosure(CLEAN)
    assert "one-sided" in text
    assert "failure to disprove" in text
    assert "NOT bit-reproducible" not in text
    # The word the whole design refuses to say about a passing check.
    assert "is reproducible" not in text


@pytest.mark.parametrize("manifest,marker", [
    (SKIPPED, "did not run"),
    (UNKNOWN, "could **not be completed**"),
    (None, "not read"),
])
def test_unknown_states_never_render_as_clean(manifest, marker):
    text = determinism_disclosure(manifest)
    assert marker in text
    assert "⚠️" in text
    assert "found no mismatch" not in text


def test_skipped_run_discloses_the_cold_matrix():
    # --no-determinism-check is not merely "less data": the matrix then runs on a cold model.
    assert "cold model" in determinism_disclosure(SKIPPED)


def _rows():
    # Same shape the report's own tests use, so these assert against a real rendered report.
    from tests.test_report import _row
    return [_row(trial=0, verdict="bypassed"), _row(trial=1, verdict="refused")]


def test_the_report_carries_the_line_and_it_is_not_buried_in_the_collapsible():
    text = build_report(_rows(), MISMATCH)
    assert "NOT bit-reproducible" in text
    # Ahead of <details>: a folded disclosure is one the reader can miss.
    assert text.index("NOT bit-reproducible") < text.index("<details>")


def test_a_report_built_without_a_manifest_says_so_rather_than_implying_clean():
    text = build_report(_rows())
    assert "not read" in text
    assert "found no mismatch" not in text


def test_the_pinning_blurb_no_longer_implies_the_pin_worked():
    text = build_report(_rows(), MISMATCH)
    assert "measured per run and stated above, not assumed here" in text


# ---------------------------------------------------------------- compare (ISC-50)

def _artifact(tmp_path, name, model, determinism):
    """A two-scenario artifact with a real manifest line, so compare reads it the way it does live."""
    import json

    from iago.artifacts import MANIFEST_RECORD

    manifest = {"record": MANIFEST_RECORD, "surface": "agentic", "model": model,
                "judge_id": "j1", "determinism": determinism}
    rows = [{"model": model, "kind": "capability", "scenario_id": f"cap{i}",
             "scenario_name": "cap", "verdict": "resisted", "floor_fired": True} for i in range(2)]
    rows.append({"model": model, "kind": "attack", "scenario_id": "s1", "scenario_name": "s1",
                 "verdict": "hijacked"})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in [manifest, *rows]))
    return p


def test_compare_warns_when_a_contributing_host_measured_itself_non_reproducible(tmp_path):
    """A delta is exactly the claim host non-determinism attacks, and compare said nothing."""
    from iago.compare import build_comparison, write_comparison_report

    a = _artifact(tmp_path, "a.jsonl", "modelA", MISMATCH["determinism"])
    b = _artifact(tmp_path, "b.jsonl", "modelB", CLEAN["determinism"])
    comp = build_comparison([a, b])
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert "Measured non-reproducible host" in text
    assert "a.jsonl" in text
    assert "b.jsonl" not in text.split("Measured non-reproducible host")[1].split("_")[0]
    assert "may be the HOST rather than a model difference" in text


def test_compare_says_unknown_is_not_clean(tmp_path):
    from iago.compare import build_comparison, write_comparison_report

    a = _artifact(tmp_path, "a.jsonl", "modelA", None)              # check skipped
    b = _artifact(tmp_path, "b.jsonl", "modelB", UNKNOWN["determinism"])
    comp = build_comparison([a, b])
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert "Reproducibility was not established" in text
    assert "a.jsonl" in text and "b.jsonl" in text
    assert "Measured non-reproducible host" not in text


def test_compare_is_silent_when_every_host_came_back_clean(tmp_path):
    from iago.compare import build_comparison, write_comparison_report

    a = _artifact(tmp_path, "a.jsonl", "modelA", CLEAN["determinism"])
    b = _artifact(tmp_path, "b.jsonl", "modelB", CLEAN["determinism"])
    text = write_comparison_report(build_comparison([a, b]), reports_dir=tmp_path).read_text()
    assert "Measured non-reproducible host" not in text
    assert "Reproducibility was not established" not in text
