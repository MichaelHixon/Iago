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
