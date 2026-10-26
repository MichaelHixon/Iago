"""A run with no model profile writes rows byte-shaped exactly as before profiles existed.

`TrialResult` carries two profile fields (`arm`, `model_profile_sha256`) for model-adaptive runs.
A `profile=None` default protects the rendered prompt bytes, not the serialized row: a plain
`asdict` would dump both as `null` into every existing artifact and every downstream reader. So the
writer omits them when unset, and this pins the pre-profile key set against a real `run()`.
"""

from dataclasses import asdict

from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.objectives import Objective
from dataclasses import fields

import pytest

from iago.runner import PROFILE_ONLY_FIELDS, TrialResult, run, trial_row
from iago.target import Target

# The row keys a chatbot artifact carried before model profiles existed (captured at ef897f8).
PRE_PROFILE_KEYS = [
    "technique_id", "technique_name", "category", "owasp", "objective_id", "objective_kind",
    "model", "seed", "temperature", "trial", "batch_id", "run_seq", "prompt", "response",
    "verdict", "confidence", "rationale", "latency_s", "timestamp", "turns", "shots", "gated",
    "decoded", "anon_response", "anon_signal", "placebo_response", "placebo_signal",
    "prov_signal", "asi", "atlas", "sink", "deadend_signal", "deadend_control", "truncated",
    "schema_version", "surface", "status", "prompt_sha256", "response_sha256", "judge_id",
    "system_prompt_sha256",
]

# The profile fields, in dataclass order, that trial_row drops from an unprofiled row.
PROFILE_KEYS = ["profile_arm", "model_profile_sha256"]


class _T(Target):
    name, is_local = "fake:model", True

    def generate(self, prompt, system=None, options=None):
        return "A" * 300

    def chat(self, messages, options=None):
        return self.generate(None)


def _result(**kw):
    base = dict(
        technique_id="t1", technique_name="Direct", category="direct-ask", owasp="LLM01",
        objective_id="o1", objective_kind="forbidden", model="fake:model", seed=1, temperature=0.8,
        trial=0, batch_id=0, run_seq=0, prompt="p", response="r", verdict="refused",
        confidence=0.9, rationale="x", latency_s=0.1, timestamp="2026-01-01T00:00:00Z",
    )
    base.update(kw)
    return TrialResult(**base)


def test_unprofiled_run_rows_keep_the_pre_profile_key_set(tmp_path):
    path = run(_T(), techniques=[Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")],
               objectives=[Objective("o1", "forbidden thing", "forbidden")], trials=1,
               artifacts_dir=tmp_path, determinism_check=False)
    _, rows = read_artifact(path)
    assert rows, "run() wrote no rows; the key-set check would pass vacuously"
    for row in rows:
        assert list(row) == PRE_PROFILE_KEYS


def test_profiled_run_stamps_every_row_and_the_manifest(tmp_path):
    """End-to-end: run(profile=p) stamps every row with profile_arm='tuned' and the profile's
    fingerprint, and records the model_profile block in the manifest. A mutation dropping either
    the row stamp or the manifest block reds here."""
    from iago.profiles import load_profiles

    prof = {p.id: p for p in load_profiles()}["llama31"]
    path = run(_T(), techniques=[Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")],
               objectives=[Objective("o1", "forbidden thing", "forbidden")], trials=1,
               artifacts_dir=tmp_path, determinism_check=False, profile=prof)
    manifest, rows = read_artifact(path)
    assert rows
    for row in rows:
        assert row["profile_arm"] == "tuned"
        assert row["model_profile_sha256"] == prof.fingerprint
    mp = manifest["model_profile"]
    assert mp["id"] == "llama31" and mp["family"] == "llama"
    assert mp["model_version"] == prof.model_version and mp["control_channel"] == prof.control_channel
    assert mp["sha256"] == prof.fingerprint
    assert mp["control_channel_class"] == "hard"  # inst-tags is a raw-token channel


def test_soft_channel_profile_is_labeled_soft_in_the_manifest(tmp_path):
    """The pooling axis travels in the artifact: an anthropic (system-role) run is marked soft, so a
    cross-family reader can refuse to pool its delta with a hard-channel run — without the S4 report."""
    from iago.profiles import load_profiles

    prof = {p.id: p for p in load_profiles()}["anthropic"]  # control_channel = system-role (soft)
    path = run(_T(), techniques=[Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")],
               objectives=[Objective("o1", "forbidden thing", "forbidden")], trials=1,
               artifacts_dir=tmp_path, determinism_check=False, profile=prof)
    manifest, _ = read_artifact(path)
    assert manifest["model_profile"]["control_channel_class"] == "soft"


def test_plain_asdict_would_leak_the_profile_fields():
    """The omission is load-bearing: without it, an unprofiled row gains two null keys."""
    leaked = [k for k in asdict(_result()) if k not in PRE_PROFILE_KEYS]
    assert leaked == list(PROFILE_ONLY_FIELDS) == PROFILE_KEYS


def test_unprofiled_row_omits_both_profile_fields_directly():
    """Direct unit pin of trial_row's omission branch, independent of the run() plumbing."""
    assert set(PROFILE_ONLY_FIELDS).isdisjoint(trial_row(_result()))


def test_profile_only_fields_are_real_dataclass_fields():
    """A rename of these fields must not silently KeyError in trial_row's del loop."""
    names = {f.name for f in fields(TrialResult)}
    assert set(PROFILE_ONLY_FIELDS) <= names


def test_hash_without_arm_fails_loud():
    """A fingerprint set without an arm would be silently dropped — assert it fails instead."""
    with pytest.raises(AssertionError, match="without profile_arm"):
        trial_row(_result(model_profile_sha256="ab" * 32))


def test_profiled_row_carries_both_fields():
    row = trial_row(_result(profile_arm="tuned", model_profile_sha256="ab" * 32))
    assert row["profile_arm"] == "tuned"
    assert row["model_profile_sha256"] == "ab" * 32


def test_baseline_arm_keeps_arm_and_a_null_profile_hash():
    """A baseline row in a profiled run says so: `profile_arm` is set, the hash is explicitly null."""
    row = trial_row(_result(profile_arm="baseline"))
    assert row["profile_arm"] == "baseline"
    assert "model_profile_sha256" in row and row["model_profile_sha256"] is None
