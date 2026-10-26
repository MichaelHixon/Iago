"""The dual-rate model-adaptive report (ISC-64): the tuned bypass rate is NEVER reported alone.

`require_dual_arms` refuses a tuned-only artifact shape (the ISC-64 raise). `write_model_adaptive_report`
renders both arms' rates and, for a seed-honoring (deterministic) family, the paired-difference CI; a
seed-dropped (stochastic) family gets separate arm rates and an OPEN delta, never a pooled CI. A single
orphaned arm-row (a mid-run error killed its twin) drops that one pair with a counted line, not a raise.
"""

import pytest

from iago.artifacts import require_dual_arms, write_model_adaptive_report


def _row(arm, *, tech="t1", obj="o1", trial=0, verdict="bypassed", kind="forbidden",
         sha="ab" * 32):
    """A minimal chatbot-surface row of one arm, shaped as `read_artifact` would return it."""
    r = {
        "technique_id": tech, "objective_id": obj, "objective_kind": kind, "trial": trial,
        "verdict": verdict, "profile_arm": arm,
        "model_profile_sha256": sha if arm == "tuned" else None,
        # is_valid_probe reads these: a forbidden row is valid unless it errored or never decoded.
        "gated": False, "decoded": None, "response": "x" * 300,
    }
    return r


def _pair(trial, *, base_verdict, tuned_verdict):
    return [_row("baseline", trial=trial, verdict=base_verdict),
            _row("tuned", trial=trial, verdict=tuned_verdict)]


DET_MANIFEST = {"sampling": {"unsupported_controls": []},
                "model_profile": {"family": "llama", "control_channel": "inst-tags",
                                  "control_channel_class": "hard"}}
STOCH_MANIFEST = {"sampling": {"unsupported_controls": ["seed"]},
                  "model_profile": {"family": "anthropic", "control_channel": "system-role",
                                    "control_channel_class": "soft"}}


# --- the ISC-64 guard ---------------------------------------------------------------------
def test_require_dual_arms_raises_on_tuned_only():
    tuned_only = [_row("tuned", trial=t) for t in range(3)]
    with pytest.raises(ValueError, match="ZERO baseline"):
        require_dual_arms(tuned_only)


def test_require_dual_arms_raises_on_baseline_only():
    with pytest.raises(ValueError, match="ZERO tuned"):
        require_dual_arms([_row("baseline", trial=t) for t in range(3)])


def test_require_dual_arms_raises_on_no_profiled_rows():
    plain = [{"profile_arm": None} for _ in range(2)]
    with pytest.raises(ValueError, match="not a model-adaptive artifact"):
        require_dual_arms(plain)


def test_require_dual_arms_accepts_both_arms():
    assert require_dual_arms(_pair(0, base_verdict="refused", tuned_verdict="bypassed")) == (1, 1)


# --- the report renders both arms (the ISC-64 probe) --------------------------------------
def test_report_renders_both_arms_and_paired_ci(tmp_path):
    rows = []
    for t in range(4):
        rows += _pair(t, base_verdict="refused", tuned_verdict="bypassed")
    path = write_model_adaptive_report(rows, manifest=DET_MANIFEST, reports_dir=tmp_path)
    text = path.read_text()
    assert "DETERMINISTIC" in text
    # Assert the COMPUTED numbers, not just static scaffolding: 4 pairs, baseline all refused (0/4),
    # tuned all bypassed (4/4), so the paired table is (both=0, raw-only=0, guarded-only=4, both=0).
    # A broken _rate_block / _paired_table / pairing key reds these, unlike a label-only assertion.
    assert "| fixed baseline | 0 | 4 |" in text
    assert "| profile-tuned | 4 | 4 |" in text
    assert "95% CI on the paired difference" in text   # the real paired_delta_line output, not the header literal
    assert "n = 4 pairs" in text and "4 guarded-only" in text
    assert "OPEN" not in text.split("## Bypass")[1].split("###")[1][:400]  # det block is not OPEN-labeled


def test_report_raises_when_baseline_arm_suppressed(tmp_path):
    # Mutation: a report over a tuned-only artifact must not render (ISC-64).
    tuned_only = [_row("tuned", trial=t, verdict="bypassed") for t in range(4)]
    with pytest.raises(ValueError, match="ZERO baseline"):
        write_model_adaptive_report(tuned_only, manifest=DET_MANIFEST, reports_dir=tmp_path)


def test_orphaned_tuned_pair_is_dropped_with_a_counted_line(tmp_path):
    # Two clean pairs + one tuned row whose baseline twin errored out mid-run. The report renders,
    # the orphan is dropped and COUNTED — not a hard raise (that would train row deletion).
    rows = _pair(0, base_verdict="refused", tuned_verdict="bypassed")
    rows += _pair(1, base_verdict="refused", tuned_verdict="refused")
    rows.append(_row("tuned", trial=2, verdict="bypassed"))  # orphan: no baseline twin
    path = write_model_adaptive_report(rows, manifest=DET_MANIFEST, reports_dir=tmp_path)
    text = path.read_text()
    assert "pair(s) excluded" in text
    assert "tuned-only: 1" in text


def test_orphaned_baseline_pair_is_dropped_with_a_counted_line(tmp_path):
    # Mirror of the tuned-orphan case for the symmetric branch: a baseline row whose tuned twin
    # errored out. The orphan_baseline count must be wired into the excluded line (a mutation
    # zeroing it reds here).
    rows = _pair(0, base_verdict="refused", tuned_verdict="bypassed")
    rows += _pair(1, base_verdict="refused", tuned_verdict="refused")
    rows.append(_row("baseline", trial=2, verdict="refused"))  # orphan: no tuned twin
    path = write_model_adaptive_report(rows, manifest=DET_MANIFEST, reports_dir=tmp_path)
    text = path.read_text()
    assert "pair(s) excluded" in text
    assert "baseline-only: 1" in text


def test_stochastic_family_gets_open_label_not_a_pooled_ci(tmp_path):
    rows = []
    for t in range(4):
        rows += _pair(t, base_verdict="refused", tuned_verdict="bypassed")
    path = write_model_adaptive_report(rows, manifest=STOCH_MANIFEST, reports_dir=tmp_path)
    text = path.read_text()
    assert "STOCHASTIC" in text
    assert "**OPEN**" in text
    # The actual paired-CI LINE (emitted by delta.paired_delta_line) must be absent — a stochastic
    # family quotes no matched-pair interval. The OPEN prose may still explain why one is withheld.
    assert "95% CI on the paired difference" not in text
    assert "Soft channel" in text              # soft control-channel class is surfaced


def test_soft_channel_warns_against_pooling(tmp_path):
    rows = _pair(0, base_verdict="refused", tuned_verdict="bypassed")
    path = write_model_adaptive_report(rows, manifest=STOCH_MANIFEST, reports_dir=tmp_path)
    assert "Do NOT pool" in path.read_text()
