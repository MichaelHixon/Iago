"""verification — the target adapter conforms and (if Ollama is up) responds."""

from types import SimpleNamespace

import pytest

from iago.target import (
    OllamaTarget,
    Target,
    _extract_content,
    available_targets,
    build_target,
)


def test_ollama_target_conforms_to_interface():
    target = OllamaTarget()
    assert isinstance(target, Target)
    assert target.name.startswith("ollama:")
    assert target.is_local is True


def test_extract_from_object_shape():
    # Regression: ollama returns a ChatResponse OBJECT, not a dict. A dict-only
    # extractor silently nulled every reply and the judge classified error strings
    # (caught 2026-07-24 only by reading a report).
    resp = SimpleNamespace(message=SimpleNamespace(content="hello from object"))
    assert _extract_content(resp) == "hello from object"


def test_extract_from_dict_shape():
    assert _extract_content({"message": {"content": "hello from dict"}}) == "hello from dict"


def test_extract_missing_content_is_none():
    assert _extract_content(SimpleNamespace(message=SimpleNamespace(content=None))) is None
    assert _extract_content({}) is None


def test_factory_builds_ollama_by_default():
    t = build_target(model="some-model")
    assert isinstance(t, OllamaTarget)
    assert t.model == "some-model"
    assert "ollama" in available_targets()


def test_factory_unknown_target_raises_with_options():
    with pytest.raises(ValueError, match="unknown target"):
        build_target("gpt-9000")


def test_ollama_generate_smoke():
    target = OllamaTarget()
    try:
        out = target.generate("Say hello in exactly three words.")
    except RuntimeError as exc:
        pytest.skip(f"Ollama not reachable: {exc}")
    assert isinstance(out, str) and len(out) > 0
