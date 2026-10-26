"""Model profiles load, validate loudly, and carry no payload bodies (ISC-62, ISC-65).

Mirrors the attack-library loader tests: every shipped profile loads; the loud-failure paths each
red under a mutation — a dropped version stamp, an unknown carrier, a duplicate id, and any
payload-body-shaped field. The no-payload scan over the shipped YAML backs ISC-65 directly.
"""

from pathlib import Path

import pytest
import yaml

from iago.config import PROFILES_DIR
from iago.profiles import (
    FORBIDDEN_BODY_FIELDS,
    MAX_DESCRIPTION_LEN,
    MAX_MARKER_LEN,
    ModelProfile,
    REQUIRED_PROFILE_FIELDS,
    load_profiles,
)

SHIPPED_IDS = {"llama31", "qwen3", "gemma", "anthropic"}


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    (tmp_path / name).write_text(yaml.safe_dump(records))
    return tmp_path


def _rec(**kw) -> dict:
    base = dict(id="p1", name="P", description="d", family="llama", model_version="3.1",
                control_channel="inst-tags", refusal_markers=["I cannot"], carriers=["none"])
    base.update(kw)
    return base


def test_every_shipped_profile_loads():
    profiles = load_profiles()
    assert {p.id for p in profiles} == SHIPPED_IDS
    assert all(isinstance(p, ModelProfile) for p in profiles)
    assert all(p.model_version for p in profiles)


def test_missing_model_version_raises(tmp_path):
    rec = _rec()
    del rec["model_version"]
    d = _write(tmp_path, "p.yaml", [rec])
    with pytest.raises(ValueError, match="missing 'model_version'"):
        load_profiles(d)


def test_unknown_carrier_raises(tmp_path):
    d = _write(tmp_path, "p.yaml", [_rec(carriers=["none", "not-a-carrier"])])
    with pytest.raises(ValueError, match="unknown carrier"):
        load_profiles(d)


def test_unknown_control_channel_raises(tmp_path):
    d = _write(tmp_path, "p.yaml", [_rec(control_channel="telepathy")])
    with pytest.raises(ValueError, match="unknown control_channel"):
        load_profiles(d)


def test_duplicate_id_raises(tmp_path):
    d = _write(tmp_path, "p.yaml", [_rec(id="dup"), _rec(id="dup", name="Q")])
    with pytest.raises(ValueError, match="duplicate profile id 'dup'"):
        load_profiles(d)


@pytest.mark.parametrize("body_field", sorted(FORBIDDEN_BODY_FIELDS))
def test_any_payload_body_field_raises(tmp_path, body_field):
    d = _write(tmp_path, "p.yaml", [_rec(**{body_field: "some operative text"})])
    with pytest.raises(ValueError, match="payload-body field"):
        load_profiles(d)


@pytest.mark.parametrize("list_field", ["refusal_markers", "carriers"])
def test_scalar_list_field_raises(tmp_path, list_field):
    """A scalar where a list belongs fails loudly, not silently as a char tuple."""
    d = _write(tmp_path, "p.yaml", [_rec(**{list_field: "not-a-list"})])
    with pytest.raises(ValueError, match=f"{list_field}.* must be a list"):
        load_profiles(d)


def test_unknown_key_raises(tmp_path):
    """A near-miss typo must raise, not load a silently-empty profile (loud-loader invariant)."""
    d = _write(tmp_path, "p.yaml", [_rec(refusal_marker=["oops-singular"])])
    with pytest.raises(ValueError, match="unknown field"):
        load_profiles(d)


def test_overlong_description_raises(tmp_path):
    """ISC-65 value guard: description too long to be a short skin (could carry a body)."""
    d = _write(tmp_path, "p.yaml", [_rec(description="x" * (MAX_DESCRIPTION_LEN + 1))])
    with pytest.raises(ValueError, match="description is"):
        load_profiles(d)


def test_overlong_refusal_marker_raises(tmp_path):
    """ISC-65 value guard: a marker long enough to carry a scaffold is not a short phrase."""
    d = _write(tmp_path, "p.yaml", [_rec(refusal_markers=["x" * (MAX_MARKER_LEN + 1)])])
    with pytest.raises(ValueError, match="refusal_marker"):
        load_profiles(d)


def test_shipped_free_text_is_within_isc65_caps():
    """Every shipped profile's free-text channels sit under the ISC-65 value caps."""
    for p in load_profiles():
        assert len(p.description) <= MAX_DESCRIPTION_LEN
        assert all(len(m) <= MAX_MARKER_LEN for m in p.refusal_markers)


def test_non_string_model_version_raises(tmp_path):
    """An unquoted `model_version: 2` parses as int and must fail at load (ISC-62), not later."""
    d = _write(tmp_path, "p.yaml", [_rec(model_version=2)])
    with pytest.raises(ValueError, match="model_version.* must be a non-empty string"):
        load_profiles(d)


def test_profiles_carry_no_payload_bodies():
    """ISC-65 scan: no shipped profile YAML entry carries any payload-body-shaped key."""
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        for rec in yaml.safe_load(path.read_text()):
            leaked = FORBIDDEN_BODY_FIELDS & set(rec)
            assert not leaked, f"{path.name}: {rec['id']!r} carries forbidden body field(s) {leaked}"


def test_fingerprint_is_stable_across_independent_loads():
    """Same profile, two independent loads → identical fingerprint (a per-row stamp must be stable)."""
    first = {p.id: p for p in load_profiles()}
    second = {p.id: p for p in load_profiles()}
    assert first["llama31"].fingerprint == second["llama31"].fingerprint


@pytest.mark.parametrize("field,value", [
    ("model_version", "9.9"),
    ("control_channel", "chatml"),
    ("family", "other"),
    ("refusal_markers", ("different",)),
    ("carriers", ("base64",)),
])
def test_fingerprint_changes_when_an_operative_field_changes(field, value):
    """ISC-62/64 stale-tuning guard: a changed version or tuning yields a DIFFERENT stamp.

    Without this, a fingerprint reduced to `id` alone passes every other test — the stamp would
    not detect stale tuning, which is its only job.
    """
    base = ModelProfile(id="p", name="P", description="d", family="llama", model_version="3.1",
                        control_channel="inst-tags", refusal_markers=("I cannot",), carriers=("none",))
    import dataclasses
    changed = dataclasses.replace(base, **{field: value})
    assert base.fingerprint != changed.fingerprint, f"fingerprint ignores {field}"
    # name/description are deliberately NOT operative — they must not move the fingerprint.
    assert base.fingerprint == dataclasses.replace(base, name="X", description="Y").fingerprint
