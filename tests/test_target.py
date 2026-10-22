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


def _models(*pairs):
    """A fake Models API client listing (id, created_at) pairs."""
    listed = [SimpleNamespace(id=i, created_at=c) for i, c in pairs]
    return SimpleNamespace(models=SimpleNamespace(list=lambda **_kw: iter(listed)))


def _api_down(**_kw):
    raise RuntimeError("401")


def test_anthropic_default_is_the_newest_model_in_the_family():
    # The default names a family, not a model, so it never goes stale: the newest Haiku wins by
    # created_at rather than list order, an old id shape still counts as the family, a lookalike
    # segment does not, and other families are ignored.
    from iago.target import AnthropicTarget

    # revert to max-by-id → 5-5 wins over the newer 10-0; drop either regex boundary → a lookalike
    # wins; drop the prerelease filter → a preview wins. Each reds this test.
    client = _models(("claude-haiku-4-5", "2025-10-01"), ("claude-opus-9", "2027-01-01"),
                     ("claude-haiku-5-5", "2026-09-01"), ("claude-haiku-10-0", "2027-06-01"),
                     ("claude-3-5-haiku-20241022", "2024-10-22"),
                     ("claude-haikuish-9", "2028-01-01"), ("claude-xhaiku-9", "2030-01-01"),
                     ("claude-haiku-6-preview", "2028-02-01"), ("claude-haiku-6-exp-0101", "2028-03-01"))
    t = AnthropicTarget(client=client)
    assert (t.model, t.name) == ("claude-haiku-10-0", "anthropic:claude-haiku-10-0")


def test_anthropic_default_counts_the_old_id_shape():
    # revert the segment match to a `claude-haiku-` prefix → the newer old-shape id is skipped.
    from iago.target import AnthropicTarget

    client = _models(("claude-3-5-haiku-x", "2029-01-01"), ("claude-haiku-4-5", "2025-10-01"))
    assert AnthropicTarget(client=client).model == "claude-3-5-haiku-x"


def test_an_explicit_anthropic_model_skips_the_lookup():
    # revert to an unconditional lookup → the failing Models API raises here.
    from iago.target import AnthropicTarget

    t = AnthropicTarget("pinned", client=SimpleNamespace(models=SimpleNamespace(list=_api_down)))
    assert t.model == "pinned"


@pytest.mark.parametrize("client", [
    _models(("claude-opus-9", "2027-01-01")),
    SimpleNamespace(models=SimpleNamespace(list=_api_down)),
])
def test_anthropic_default_refuses_when_no_model_can_be_found(client):
    from iago.target import AnthropicTarget

    with pytest.raises(ValueError, match="pass --model"):
        AnthropicTarget(client=client)


def test_cli_run_without_model_builds_the_anthropic_default(monkeypatch):
    # `run --target anthropic` with no --model hands None to the target, which looks up its own
    # default, never an Ollama tag.
    import iago.target
    from iago import cli

    looked_up, ran = [], []

    def stop(target, **_kw):
        ran.append(target.name)
        raise SystemExit

    monkeypatch.setattr(iago.target, "latest_model", lambda client, family: looked_up.append(family) or "newest")
    monkeypatch.setattr(cli, "run", stop)
    with pytest.raises(SystemExit):
        cli.main(["run", "--target", "anthropic", "--authorized", "--smoke"])
    assert (looked_up, ran) == (["haiku"], ["anthropic:newest"])


@pytest.mark.parametrize("argv", [["run"], ["defense-delta"], ["adaptive-run"]])
def test_authorized_api_target_reaches_the_lookup(monkeypatch, argv):
    # revert any command's early check to ignore --authorized → it refuses and never looks up.
    import iago.target
    from iago import cli

    class _Built(Exception):
        pass

    looked_up = []

    def lookup(client, family):
        looked_up.append(family)
        raise _Built

    monkeypatch.setattr(iago.target, "latest_model", lookup)
    with pytest.raises(_Built):
        cli.main([*argv, "--target", "anthropic", "--authorized"])
    assert looked_up == ["haiku"]


def test_regrade_defaults_to_the_calibrated_judge():
    # The CLI default must be the judge the published accuracy was measured on.
    from iago import cli
    from iago.judge_claude import DEFAULT_JUDGE_MODEL

    assert cli.build_parser().parse_args(["regrade", "x"]).judge_model == DEFAULT_JUDGE_MODEL


@pytest.mark.parametrize("argv", [["run"], ["defense-delta"], ["adaptive-run"]])
def test_unauthorized_api_target_refuses_before_any_lookup(monkeypatch, capsys, argv):
    # No --authorized means nothing reaches the API, not even the default's model lookup.
    import iago.target
    from iago import cli

    def must_not_look_up(*_a, **_kw):
        raise AssertionError("looked up a model before the authorization check")

    monkeypatch.setattr(iago.target, "latest_model", must_not_look_up)
    assert cli.main([*argv, "--target", "anthropic"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_api_targets_matches_each_backends_is_local():
    # API_TARGETS restates `is_local` by kind so the CLI can refuse before building; a new non-local
    # backend that is not listed would skip the early refusal, so the two must agree.
    from iago.replay import ReplayTarget
    from iago.target import API_TARGETS, AnthropicTarget, OllamaTarget

    classes = {"ollama": OllamaTarget, "anthropic": AnthropicTarget, "replay": ReplayTarget}
    assert set(classes) == set(available_targets(include_run_only=True))
    assert API_TARGETS == {kind for kind, cls in classes.items() if not cls.is_local}


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
    return AnthropicTarget("m", client=SimpleNamespace(messages=msgs)), msgs


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
