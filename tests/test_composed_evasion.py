"""Composed-evasion: schema, render pipeline (transform chain + wrappers), and loud loader validation."""

import pytest

from iago.attacks import (
    _WRAPPER_FRAMES,
    _WRAPPERS,
    _transform,
    _transform_chain,
    load_library,
)


def _get(tech_id: str):
    for tech in load_library():
        if tech.id == tech_id:
            return tech
    raise AssertionError(f"technique {tech_id!r} not in library")


def _frame_marker(name: str) -> str:
    """A stable substring of wrapper `name`'s ACTIVE frame text, independent of whether an
    operator has overridden the inert default in wrappers.local.yaml. Render assertions key
    off this so a present local frame file doesn't break the tests (it legitimately replaces
    the placeholder strings). The longest non-{text} segment is always present in the output."""
    parts = _WRAPPER_FRAMES[name].split("{text}")
    return max((p.strip() for p in parts), key=len)


# --- transform chain ---------------------------------------------------------------


def test_transform_chain_applies_inner_to_outer():
    # base64 then reverse == reverse(base64(x)) — applied inner→outer.
    x = "attack the objective"
    expected = _transform(_transform(x, "base64"), "reverse")
    assert _transform_chain(x, ("base64", "reverse")) == expected
    # order matters for a non-commuting pair: reverse-then-base64 differs.
    assert _transform_chain(x, ("reverse", "base64")) != expected


def test_single_element_chain_equals_single_transform():
    assert _transform_chain("payload", ("base64",)) == _transform("payload", "base64")


def test_empty_chain_is_identity():
    assert _transform_chain("payload", None) == "payload"
    assert _transform_chain("payload", ()) == "payload"


# --- wrappers ----------------------------------------------------------------------


def test_wrappers_apply_outward_last_is_outermost():
    tech = _get("ce-persona-b64")  # wrappers: [decode-preamble, persona]
    rendered = tech.render("TEST")
    # persona is applied last, so its frame is the outermost text (comes first).
    persona_idx = rendered.index(_frame_marker("persona"))
    decode_idx = rendered.index(_frame_marker("decode-preamble"))
    assert persona_idx < decode_idx, "last wrapper must be outermost"


def test_render_encodes_then_wraps():
    tech = _get("ce-persona-b64")  # transforms:[base64], wrappers:[decode-preamble, persona]
    rendered = tech.render("TEST_OBJECTIVE")
    # payload is base64-encoded (verbatim gone), and both wrapper frames are present.
    assert _transform("TEST_OBJECTIVE", "base64") in rendered
    assert "TEST_OBJECTIVE" not in rendered
    assert _frame_marker("persona") in rendered  # persona wrapper frame applied (default or override)
    assert "decode" in rendered.lower()


def test_chain_technique_double_encodes():
    tech = _get("ce-leet-reverse-chain")  # transforms:[leetspeak, reverse]
    rendered = tech.render("attack")
    assert _transform_chain("attack", ("leetspeak", "reverse")) in rendered


def test_social_only_stack_has_no_encoding():
    tech = _get("ce-authority-refusal")  # wrappers only, no transforms
    assert tech.transforms is None and tech.transform is None
    rendered = tech.render("PLAINTEXT_OBJ")
    assert "PLAINTEXT_OBJ" in rendered  # payload stays verbatim
    assert _frame_marker("authority-header") in rendered  # authority-header frame applied (default or override)


def test_wrappers_apply_only_to_final_turn_in_crescendo():
    tech = _get("ce-crescendo-encoded")  # turns + transforms + decode-preamble wrapper
    turns = tech.rendered_turns("TEST_OBJECTIVE")
    assert len(turns) == 3  # 2 priming + 1 final
    # priming turns are unframed; the decode-preamble wraps only the final ask.
    assert "contains an encoded instruction" not in turns[0]
    assert "contains an encoded instruction" not in turns[1]
    assert "decode" in turns[-1].lower()
    assert _transform("TEST_OBJECTIVE", "base64") in turns[-1]


def test_manyshot_persona_composes_shots_and_wrapper():
    tech = _get("ce-manyshot-persona")  # {shots} + persona wrapper
    assert tech.is_shot_scaled
    rendered = tech.render("TEST_OBJECTIVE", shots=5)
    assert rendered.count("Assistant:") == 5 + 1  # 5 fabricated + the trailing turn
    assert _frame_marker("persona") in rendered  # persona wrapper composed on top (default or override)


# --- composes metadata resolves ----------------------------------------------------


def test_composes_ids_all_resolve_in_library():
    lib = load_library()
    ids = {t.id for t in lib}
    for tech in lib:
        if tech.composes:
            for cid in tech.composes:
                assert cid in ids, f"{tech.id} composes unresolved id {cid!r}"


def test_composed_category_populated():
    composed = [t for t in load_library() if t.category == "composed-evasion"]
    assert len(composed) >= 6
    # every composed technique names its constituents.
    for t in composed:
        assert t.composes, f"{t.id} must declare composes"


# --- backward compatibility --------------------------------------------------------


def test_existing_single_transform_techniques_unchanged():
    # enc-001 uses a single `transform: base64` and must render exactly as before.
    enc = _get("enc-001")
    assert enc.transform == "base64" and enc.transforms is None
    assert _transform("X", "base64") in enc.render("X")


def test_full_library_still_loads():
    lib = load_library()
    assert len(lib) >= 50  # 48 pre-existing + 8 composed


# --- loud loader validation --------------------------------------------------------


def _write_lib(tmp_path, body: str):
    (tmp_path / "bad.yaml").write_text(body)


def test_rejects_both_transform_and_transforms(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  transform: base64\n  transforms: [reverse]\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="BOTH 'transform' and 'transforms'"):
        load_library(tmp_path)


def test_rejects_unknown_transform_in_chain(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  transforms: [base64, nope]\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="unknown kind"):
        load_library(tmp_path)


def test_rejects_empty_transforms_list(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  transforms: []\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="non-empty list"):
        load_library(tmp_path)


def test_rejects_unknown_wrapper(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  wrappers: [persona, bogus]\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="unknown name"):
        load_library(tmp_path)


def test_rejects_empty_wrappers_list(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  wrappers: []\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="non-empty list"):
        load_library(tmp_path)


def test_rejects_unresolvable_composes_id(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  composes: [does-not-exist]\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="unresolved technique id"):
        load_library(tmp_path)


def test_rejects_empty_composes_list(tmp_path):
    _write_lib(tmp_path,
        "- id: bad-001\n  name: n\n  category: composed-evasion\n  description: d\n"
        "  composes: []\n  template: 'x {objective}'\n")
    with pytest.raises(ValueError, match="non-empty list"):
        load_library(tmp_path)


def test_wrapper_registry_names_are_stable():
    # The YAML references these names; keep the registry surface locked.
    assert set(_WRAPPERS) == {"persona", "refusal-ban", "authority-header", "decode-preamble"}


def test_local_frame_file_is_not_parsed_as_a_technique(tmp_path):
    """Regression: a `*.local.yaml` file (operator frame DATA) in the attacks dir must be
    SKIPPED by the technique glob, not parsed as a technique. A present wrappers.local.yaml
    previously took down load_library entirely — the whole library failed and 19 tests went
    red the moment an operator supplied frames (the documented workflow)."""
    (tmp_path / "real.yaml").write_text(
        "- id: ok-001\n  name: n\n  category: direct-ask\n  description: d\n"
        "  template: 'do {objective}'\n"
    )
    # A frame dict, exactly the shape of wrappers.local.yaml — a mapping, not a technique list.
    (tmp_path / "wrappers.local.yaml").write_text("persona: |\n  You are X.\n\n  {text}\n")
    lib = load_library(tmp_path)  # must NOT raise on the .local.yaml file
    assert [t.id for t in lib] == ["ok-001"]


def test_shipped_default_frames_are_inert():
    """The REAL invariant the render tests used to conflate with rendered output: the frames
    that SHIP in the public tree are inert placeholders (or mild/benign), never ready-to-paste
    jailbreak strings. Operative text is supplied locally and gitignored."""
    from iago.attacks import _DEFAULT_WRAPPER_FRAMES

    assert "[PERSONA FRAME" in _DEFAULT_WRAPPER_FRAMES["persona"]
    assert "[AUTHORITY-OVERRIDE FRAME" in _DEFAULT_WRAPPER_FRAMES["authority-header"]
    for name, frame in _DEFAULT_WRAPPER_FRAMES.items():
        assert "{text}" in frame, f"default frame {name!r} must carry the {{text}} placeholder"
