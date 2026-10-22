"""verification — the target adapter conforms and (if Ollama is up) responds."""

import os
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
    target = OllamaTarget("some-model")
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


def test_factory_none_model_means_backend_default():
    # cli passes model=None when --model is omitted; a builder with its own default resolves
    # None to it rather than carry None into the target.
    from iago.target import DEFAULT_ANTHROPIC_MODEL

    assert build_target("anthropic", model=None).model == DEFAULT_ANTHROPIC_MODEL


@pytest.mark.parametrize("model", [None, "", "  "])
def test_factory_ollama_has_no_default_model(model):
    # Ollama deliberately has no default: an omitted tag refuses and names the flag.
    with pytest.raises(ValueError, match="no default model; pass --model"):
        build_target("ollama", model=model)


@pytest.mark.parametrize("model", [None, ""])
def test_ollama_leaves_refuse_a_missing_model(model):
    # The refusal lives in the two constructors, so a library caller that skips build_target
    # cannot run a whole matrix of errors labeled `ollama:None`.
    from iago.agent_run import ollama_chat_fn

    with pytest.raises(ValueError, match="no default model"):
        OllamaTarget(model)
    with pytest.raises(ValueError, match="no default model"):
        ollama_chat_fn(model)


def test_factory_unknown_target_raises_with_options():
    with pytest.raises(ValueError, match="unknown target"):
        build_target("gpt-9000")


def test_ollama_generate_smoke():
    # Live and opt-in: Ollama has no default model, so name a pulled one in IAGO_SMOKE_MODEL.
    model = os.environ.get("IAGO_SMOKE_MODEL")
    if not model:
        pytest.skip("set IAGO_SMOKE_MODEL to a pulled Ollama tag to run the live smoke")
    target = OllamaTarget(model)
    try:
        out = target.generate("Say hello in exactly three words.")
    except RuntimeError as exc:
        pytest.skip(f"Ollama not reachable: {exc}")
    assert isinstance(out, str) and len(out) > 0


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])


def _anthropic():
    from iago.target import AnthropicTarget

    msgs = _FakeMessages()
    return AnthropicTarget(client=SimpleNamespace(messages=msgs)), msgs


def test_anthropic_declares_seed_unsupported():
    from iago.target import unsupported_options

    t, _ = _anthropic()
    assert unsupported_options(t, {"temperature": 0.0, "seed": 1, "num_predict": 8}) == ["seed"]
    assert unsupported_options(OllamaTarget("some-model"), {"seed": 1, "anything": 2}) == []


def test_anthropic_maps_num_predict_to_max_tokens():
    t, msgs = _anthropic()
    t.generate("hi", options={"num_predict": 160})
    t.chat([{"role": "user", "content": "hi"}], options={"num_predict": 40})
    t.generate("hi")
    assert [c["max_tokens"] for c in msgs.calls] == [160, 40, 1024]


def test_anthropic_chat_passes_single_system_message():
    t, msgs = _anthropic()
    t.chat([{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}])
    assert msgs.calls[0]["system"] == "be terse"
    assert msgs.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_chat_rejects_second_system_message():
    # Overwriting silently dropped the first system message's instructions.
    t, msgs = _anthropic()
    with pytest.raises(ValueError, match="at most one system message"):
        t.chat([{"role": "system", "content": "a"}, {"role": "user", "content": "hi"},
                {"role": "system", "content": "b"}])
    assert msgs.calls == []


def test_anthropic_generate_passes_system_and_temperature():
    t, msgs = _anthropic()
    t.generate("hi", system="be terse", options={"temperature": 0.0})
    t.generate("hi")
    assert msgs.calls[0]["system"] == "be terse"
    assert msgs.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert msgs.calls[0]["temperature"] == 0.0  # 0.0 so an `or 1.0` slip goes red
    assert msgs.calls[1]["system"] == ""
