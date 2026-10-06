"""OllamaTarget anti-runaway: every single-turn call carries a token cap and a wall-clock timeout.

A llama3.2:1b matrix run hung indefinitely on 2026-10-05: the model fell into a repetition loop
and OllamaTarget set neither bound. These tests pin both bounds, the caller override, the
truncation flag, and the determinism probe's own 160-token cap (which is not a truncation).
Each test names the revert that turns it red.
"""

from types import SimpleNamespace

import pytest

from iago.config import DEFAULT_TARGET_GEN_TIMEOUT, DEFAULT_TARGET_NUM_PREDICT
from iago.runner import load_artifacts, run
from iago.target import OllamaTarget, TruncatedReply, generation_bounds


class _FakeClient:
    init: dict = {}
    options: dict = {}
    done_reason: str = "stop"
    as_object: bool = False  # real ollama returns a ChatResponse object, not a dict
    raise_exc: Exception | None = None

    def __init__(self, **kwargs):
        _FakeClient.init = kwargs

    def chat(self, *, model, messages, options):
        _FakeClient.options = options
        if _FakeClient.raise_exc is not None:
            raise _FakeClient.raise_exc
        if _FakeClient.as_object:
            return SimpleNamespace(message=SimpleNamespace(content="ok"), done_reason=_FakeClient.done_reason)
        return {"message": {"content": "ok"}, "done_reason": _FakeClient.done_reason}


@pytest.fixture
def fake(monkeypatch):
    import ollama

    _FakeClient.init, _FakeClient.options = {}, {}
    _FakeClient.done_reason, _FakeClient.as_object, _FakeClient.raise_exc = "stop", False, None
    monkeypatch.setattr(ollama, "Client", _FakeClient)
    return _FakeClient


ENTRY_POINTS = {
    "generate": lambda t, opts=None: t.generate("hi", options=opts),
    "chat": lambda t, opts=None: t.chat([{"role": "user", "content": "hi"}], options=opts),
}


def test_defaults_are_sane_ceilings():
    # revert: a default of 0 or a runaway-sized cap → fails
    assert 0 < DEFAULT_TARGET_NUM_PREDICT <= 8192
    assert 0 < DEFAULT_TARGET_GEN_TIMEOUT <= 900


@pytest.mark.parametrize("call", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_both_entry_points_apply_timeout_and_token_cap(fake, call):
    # revert: drop the setdefault (no cap) or the Client timeout kwarg → fails
    assert call(OllamaTarget("m"), {"seed": 1}) == "ok"
    assert fake.init == {"timeout": DEFAULT_TARGET_GEN_TIMEOUT}
    assert fake.options == {"seed": 1, "num_predict": DEFAULT_TARGET_NUM_PREDICT}


def test_caller_options_are_not_mutated(fake):
    # revert: write the cap into the caller's dict instead of a copy → fails (the runner reuses it)
    opts = {"seed": 1}
    OllamaTarget("m").generate("hi", options=opts)
    assert opts == {"seed": 1}


def test_caller_num_predict_wins(fake):
    # revert: always overwrite num_predict with the default → fails
    OllamaTarget("m").generate("hi", options={"num_predict": 160})
    assert fake.options["num_predict"] == 160


def test_timeout_zero_disables_the_wall_clock_bound(fake):
    OllamaTarget("m", timeout=0).generate("hi")
    assert fake.init == {}


@pytest.mark.parametrize("as_object", [False, True], ids=["dict", "ChatResponse-object"])
@pytest.mark.parametrize("call", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_own_cap_hit_returns_a_flagged_reply_that_is_still_judged(fake, call, as_object):
    # revert: drop the TruncatedReply wrap, or read done_reason only from a dict → fails
    fake.done_reason, fake.as_object = "length", as_object
    reply = call(OllamaTarget("m"))
    assert reply == "ok" and isinstance(reply, TruncatedReply)


def test_a_cap_the_caller_asked_for_is_not_a_truncation(fake):
    # The determinism probe caps at 160 on purpose; flagging (or raising on) that cap would
    # null its result on every run. revert: drop the caller_cap check → fails
    fake.done_reason = "length"
    reply = OllamaTarget("m").generate("hi", options={"num_predict": 160})
    assert reply == "ok" and not isinstance(reply, TruncatedReply)


def test_a_timeout_surfaces_as_a_loud_runtime_error(fake):
    # Wrapping only: the bound itself is httpx's, pinned by the Client(timeout=) assertion above.
    import httpx

    fake.raise_exc = httpx.ReadTimeout("timed out")
    with pytest.raises(RuntimeError, match=rf"gen timeout={DEFAULT_TARGET_GEN_TIMEOUT}s"):
        OllamaTarget("m").generate("hi")


def test_generation_bounds_reads_through_a_guard_wrapper():
    inner = OllamaTarget("m", num_predict=99, timeout=5)
    assert generation_bounds(SimpleNamespace(inner=inner)) == {"target_num_predict": 99, "target_gen_timeout": 5}
    assert generation_bounds(SimpleNamespace(name="anthropic")) is None
    # Only a real OllamaTarget records bounds: a look-alike with the attributes is not one, so the
    # recorded cap and timeout always come from the backend that enforces them.
    # revert: isinstance → hasattr in generation_bounds → fails
    assert generation_bounds(SimpleNamespace(num_predict=1, timeout=2)) is None


def test_runner_flags_a_truncated_row_and_records_the_bounds(fake, tmp_path):
    # End to end through the real runner: a capped reply is judged (status evaluated) and flagged.
    # revert: drop the row's truncated= or the manifest's generation_bounds → fails
    from iago.attacks import Technique
    from iago.objectives import Objective

    fake.done_reason = "length"
    tech = Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")
    obj = Objective("o1", "forbidden thing", "forbidden")
    out = run(OllamaTarget("m"), trials=1, artifacts_dir=tmp_path, techniques=[tech], objectives=[obj],
              determinism_check=False)
    rows = load_artifacts(out)
    assert [r["truncated"] for r in rows] == [True]
    assert rows[0]["status"] == "evaluated"
    manifest = __import__("json").loads(out.read_text().splitlines()[0])
    assert manifest["sampling"]["generation_bounds"] == {
        "target_num_predict": DEFAULT_TARGET_NUM_PREDICT, "target_gen_timeout": DEFAULT_TARGET_GEN_TIMEOUT}
