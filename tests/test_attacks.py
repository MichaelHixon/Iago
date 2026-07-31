"""verification — the attack library is well-formed and meets the count gate."""

from iago.attacks import load_library, summarize
from iago.config import CATEGORIES


def test_library_has_at_least_15_techniques():
    lib = load_library()
    assert len(lib) >= 15, f"expected >=15 techniques, got {len(lib)}"


def test_all_categories_populated():
    counts = summarize(load_library())
    for category in CATEGORIES:
        assert counts[category] >= 1, f"category '{category}' has no techniques"


def test_every_template_has_placeholder_and_renders():
    for tech in load_library():
        assert "{objective}" in tech.template
        rendered = tech.render("TEST_OBJECTIVE")
        assert "{objective}" not in rendered
        if tech.transform in (None, "none"):
            # Non-transforming techniques inject the objective verbatim.
            assert "TEST_OBJECTIVE" in rendered
        else:
            # Transforming techniques encode the payload, so the verbatim text is
            # gone but the encoded form must be present (real evasion, the encoding path).
            from iago.attacks import _transform
            assert _transform("TEST_OBJECTIVE", tech.transform) in rendered


def test_technique_ids_are_unique():
    ids = [t.id for t in load_library()]
    assert len(ids) == len(set(ids)), "duplicate technique ids found"


def test_cipher_transforms_encode_correctly():
    """The Tier-1 cipher transforms actually encipher the payload (not plain text)."""
    from iago.attacks import _transform

    assert _transform("HELLO", "rot13") == "URYYB"
    assert _transform("abc", "caesar") == "def"
    assert _transform("SOS", "morse") == "... --- ..."
    # Round-trip sanity: ROT13 is its own inverse.
    assert _transform(_transform("attack", "rot13"), "rot13") == "attack"


def _get(tech_id: str):
    for tech in load_library():
        if tech.id == tech_id:
            return tech
    raise AssertionError(f"technique {tech_id!r} not in library")


def test_new_round2_categories_present():
    """Competing-objectives and template-injection techniques loaded and render."""
    for tid in ("co-001", "co-002", "ti-001", "ti-002"):
        tech = _get(tid)
        rendered = tech.render("TEST_OBJECTIVE")
        assert "TEST_OBJECTIVE" in rendered
    # template-injection forges chat-template control tokens.
    assert "<|im_start|>" in _get("ti-001").render("X")
    assert "<|start_header_id|>" in _get("ti-002").render("X")


def test_many_shot_scaling_is_runner_driven():
    """ms-001 expands {shots} to N fabricated compliant pairs; the count is drivable."""
    from iago.attacks import _MANY_SHOT_POOL

    ms = _get("ms-001")
    assert ms.is_shot_scaled
    assert ms.shots == 8  # baseline default from YAML

    default_render = ms.render("TEST_OBJECTIVE")
    assert "{shots}" not in default_render
    assert "TEST_OBJECTIVE" in default_render
    assert default_render.count("Assistant:") == 8 + 1  # 8 fabricated + the trailing turn

    # The runner can override the shot count to exercise long-context scaling.
    scaled = ms.render("TEST_OBJECTIVE", shots=64)
    assert scaled.count("Assistant:") == 64 + 1
    assert len(scaled) > len(default_render)

    # shot_count reports the effective count; None for non-scaled techniques.
    assert ms.shot_count() == 8
    assert ms.shot_count(64) == 64
    assert _get("co-001").shot_count(64) is None  # override ignored when not shot-scaled


def test_build_shots_is_deterministic_and_cycles():
    from iago.attacks import _build_shots, _MANY_SHOT_POOL

    assert _build_shots(3) == _build_shots(3)  # deterministic
    n = len(_MANY_SHOT_POOL)
    big = _build_shots(n + 2)
    assert big.count("User:") == n + 2  # cycles past the pool length
    import pytest
    with pytest.raises(ValueError):
        _build_shots(0)


def test_shots_placeholder_requires_int_default(tmp_path):
    """A {shots} template with no integer 'shots' default fails loudly."""
    import pytest

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "- id: bad-001\n"
        "  name: Missing shots default\n"
        "  category: many-shot\n"
        "  description: x\n"
        "  template: 'prefix {shots} then {objective}'\n"
    )
    with pytest.raises(ValueError, match="shots"):
        load_library(tmp_path)


def test_shots_rejects_bool_and_nonpositive(tmp_path):
    """A {shots} default that is a bool or < 1 fails loudly (fail-loud loader ethos)."""
    import pytest

    def _write(val: str) -> None:
        (tmp_path / "bad.yaml").write_text(
            "- id: bad-001\n"
            "  name: Bad shots default\n"
            "  category: many-shot\n"
            "  description: x\n"
            f"  shots: {val}\n"
            "  template: 'prefix {shots} then {objective}'\n"
        )

    for val in ("true", "0", "-3"):
        _write(val)
        with pytest.raises(ValueError, match="positive integer"):
            load_library(tmp_path)
