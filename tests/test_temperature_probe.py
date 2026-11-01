"""Claude 5.5 rejects `temperature` (ISC-143..147).

`AnthropicTarget` learns per model, by one 1-token probe when built, whether `temperature` is
accepted; a model that refuses it runs at its default with the manifest and every report saying so.
Fakes are built from the 400 body the API returned on 2026-11-01 (`test_target._bad_request`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iago.adaptive import NextMove, run_adaptive_conversation
from iago.agent_run import run_agent_suite
from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.campaign import Campaign, SurfaceResult, write_campaign_report
from iago.compare import SAMPLING_MISMATCH_LABEL, build_comparison, write_comparison_report
from iago.objectives import Objective
from iago.report import determinism_disclosure
from iago.runner import _determinism_probe, run
from iago.target import (AnthropicTarget, SamplingRefused, Target, is_temperature_refusal,
                         unsupported_options)
from iago.trigger import effective_trials, render_trigger_report

from test_adaptive import FORBIDDEN, _ScriptAttacker
from test_delivery_signal import BASE, scripted_chat
from test_target import DEPRECATED, _anthropic, _bad_request, _refuse_any_temperature
from test_trigger import _run as _run_trigger


# --- ISC-143: the build-time probe ----------------------------------------------------------

def test_anthropic_drops_temperature_the_model_refuses():
    """Mutations: ignore the probe result, or still send `temperature` (1.0 or the option)."""
    t, msgs = _anthropic(refuse=_refuse_any_temperature)
    assert unsupported_options(t, {"temperature": 0.8, "seed": 1}) == ["seed", "temperature"]
    t.generate("hi", options={"temperature": 0.8})
    t.chat([{"role": "user", "content": "hi"}], options={"temperature": 0.8})
    assert [("temperature" in c) for c in msgs.real] == [False, False]


def test_anthropic_keeps_temperature_the_model_accepts():
    """Mutation: drop `temperature` unconditionally (option B creeping in)."""
    t, msgs = _anthropic()
    assert unsupported_options(t, {"temperature": 0.8, "seed": 1}) == ["seed"]
    t.generate("hi", options={"temperature": 0.8})
    assert msgs.real[0]["temperature"] == 0.8


def test_anthropic_probe_sends_a_non_default_temperature():
    """A model that accepts only the default 1.0. Mutation: probe at 1.0 -> nothing is dropped."""
    t, msgs = _anthropic(refuse=lambda kw: _bad_request(DEPRECATED)
                         if kw.get("temperature", 1.0) != 1.0 else None)
    assert "temperature" not in t.supported_options
    assert msgs.calls[0]["temperature"] != 1.0


def test_a_temperature_validation_400_fails_the_build():
    """A range 400 is not a deprecation. Mutation: classify "400 + temperature" alone as dropped."""
    with pytest.raises(ValueError, match="could not probe m for sampling support"):
        _anthropic(refuse=lambda kw: _bad_request("temperature: range: 0 <= x <= 1"))


def test_anthropic_probe_failure_fails_the_build():
    """Mutation: widen the except clause to return the declared set on any error."""
    with pytest.raises(ValueError, match="could not probe m .*InternalServerError"):
        _anthropic(refuse=lambda kw: _bad_request("overloaded", status=500, kind="api_error"))


def test_anthropic_probe_is_one_call():
    """Mutation: probe inside `_create` (lazy discovery) -> one probe per call."""
    t, msgs = _anthropic()
    t.generate("hi")
    t.chat([{"role": "user", "content": "hi"}])
    probes = [c for c in msgs.calls if c["max_tokens"] == 1]
    assert len(probes) == 1 and msgs.calls[0] is probes[0]


def test_a_refused_probe_still_counts_as_accepted():
    """A probe answered 200 with `stop_reason: refusal` did not reject the parameter.
    Mutation: treat any non-end_turn probe reply as a drop."""
    t, _ = _anthropic(reply=lambda: SimpleNamespace(content=[], stop_reason="refusal",
                                                    stop_details=None, model="m"))
    assert "temperature" in t.supported_options


def test_classifier_reads_the_captured_shape_only():
    assert is_temperature_refusal(_bad_request(DEPRECATED))
    assert not is_temperature_refusal(_bad_request("`top_p` is deprecated for this model."))
    assert not is_temperature_refusal(_bad_request(DEPRECATED, kind="authentication_error"))
    assert not is_temperature_refusal(RuntimeError(DEPRECATED))


# --- ISC-144: no collapse at temperature 0 when temperature was dropped ----------------------

class _Plain(Target):
    name, is_local = "fake:model", True

    def __init__(self, supported):
        self.supported_options = frozenset(supported)

    def generate(self, prompt, system=None, options=None):
        return "Sure. Here is a short, harmless answer to that."


def test_effective_trials_reads_the_dropped_list():
    """Mutation: drop the `dropped` guard -> 1."""
    assert effective_trials(3, 0, ["seed", "temperature"]) == 3
    assert effective_trials(3, 0, ["seed"]) == 1
    assert effective_trials(3, 0.8, []) == 3


def test_no_collapse_when_the_target_drops_temperature(tmp_path):
    manifest, rows = read_artifact(_run_trigger(_Plain(set()), tmp_path, trials=3, temperature=0))
    assert manifest["sampling"]["trials"] == 3
    assert {r["trial"] for r in rows} == {0, 1, 2}


def test_collapse_still_happens_when_temperature_is_accepted(tmp_path):
    manifest, rows = read_artifact(_run_trigger(_Plain({"temperature", "seed"}), tmp_path,
                                                trials=3, temperature=0))
    assert (manifest["sampling"]["trials"], manifest["sampling"]["trials_requested"]) == (1, 3)
    assert {r["trial"] for r in rows} == {0}


# --- ISC-145: a mid-run temperature 400 fails the run --------------------------------------

def _refuses_after_probe():
    """Accepts the probe, then refuses `temperature` on the next call (the API changed under us)."""
    seen = []

    def refuse(kw):
        seen.append(kw)
        return _bad_request(DEPRECATED) if len(seen) > 1 and "temperature" in kw else None
    t, _ = _anthropic(refuse=refuse)
    assert "temperature" in t.supported_options
    return t


TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


@pytest.mark.parametrize("determinism_check", [True, False])
def test_temperature_refusal_mid_run_fails_run_and_defense_delta(tmp_path, determinism_check):
    """`run` and `defense-delta` (which calls `run` per arm). Mutation: derive SamplingRefused
    from RuntimeError only -> RUN-ERROR rows and the run completes."""
    with pytest.raises(SamplingRefused):
        run(_refuses_after_probe(), trials=1, authorized=True, artifacts_dir=tmp_path,
            techniques=TECHS, objectives=OBJS, determinism_check=determinism_check)


def test_temperature_refusal_mid_run_fails_adaptive_run():
    with pytest.raises(SamplingRefused):
        run_adaptive_conversation(_refuses_after_probe(), FORBIDDEN,
                                  _ScriptAttacker([NextMove("hi", "ask", "Ask")]), max_turns=1)


def test_temperature_refusal_mid_run_fails_trigger_check(tmp_path):
    with pytest.raises(SamplingRefused):
        _run_trigger(_refuses_after_probe(), tmp_path, trials=1)


def test_determinism_probe_reraises_run_fatal():
    """Mutation: drop the probe's `except RunFatal: raise` -> filed as probe metadata."""
    with pytest.raises(SamplingRefused):
        _determinism_probe(_refuses_after_probe(), temperature=0.8, base_seed=1)


# --- ISC-146: every reader says temperature was not applied --------------------------------

DROPPED = {"sampling": {"temperature": 0.8, "trials": 1, "base_seed": 1,
                        "unsupported_controls": ["seed", "temperature"]}}


def test_pairing_note_names_both_dropped_controls():
    text = render_trigger_report([], DROPPED | {"model": "anthropic:m"})
    assert "matched on prompt only" in text
    assert "temperature 0.8 (requested; not applied by the target)" in text


def test_pairing_note_unchanged_when_only_the_seed_is_dropped():
    text = render_trigger_report([], {"model": "m", "sampling": {"temperature": 0.8,
                                                                 "unsupported_controls": ["seed"]}})
    assert "matched on prompt and temperature only" in text
    assert "not applied by the target" not in text


def test_sampling_line_marks_a_dropped_temperature():
    line = determinism_disclosure(DROPPED)
    assert ("Temperature-pinned reproducibility does not hold on this model: the requested 0.8 "
            "was not applied") in line
    assert "Temperature-pinned" not in determinism_disclosure(
        {"sampling": {"temperature": 0.8, "unsupported_controls": ["seed"]}})


# --- ISC-147: compare and campaign refuse unequal sampling unless asked --------------------

def _agent_artifact(tmp_path, name, honors):
    def fn(messages, tools, options):
        return scripted_chat(["done"])(messages, tools, options)
    fn.supported_options = honors
    return run_agent_suite(fn, model_name=f"fake:{name}", trials=1, max_steps=2, scenarios=[BASE],
                           artifacts_dir=tmp_path / name)


def _unequal(tmp_path):
    return (_agent_artifact(tmp_path, "default", frozenset()),
            _agent_artifact(tmp_path, "pinned", frozenset({"temperature", "seed"})))


def test_compare_refuses_unequal_sampling(tmp_path):
    """Mutation: remove the refusal -> no error."""
    default, pinned = _unequal(tmp_path)
    with pytest.raises(ValueError, match="--allow-sampling-mismatch") as exc:
        build_comparison([default, pinned])
    assert default.name in str(exc.value) and pinned.name in str(exc.value)


def test_compare_cli_exits_2_on_unequal_sampling(tmp_path, capsys):
    from iago import cli
    default, pinned = _unequal(tmp_path)
    assert cli.main(["compare", str(default), str(pinned)]) == 2
    assert "allow-sampling-mismatch" in capsys.readouterr().err


def test_compare_labels_unequal_sampling_when_allowed(tmp_path):
    """Mutation: drop the label -> absent from the heading and the caveat."""
    default, pinned = _unequal(tmp_path)
    comp = build_comparison([default, pinned], allow_sampling_mismatch=True)
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert f"## Difference between models: {SAMPLING_MISMATCH_LABEL}" in text
    assert "allow-sampling-mismatch" in text


def test_compare_names_the_default_temperature_arm(tmp_path):
    default, pinned = _unequal(tmp_path)
    comp = build_comparison([default, pinned], allow_sampling_mismatch=True)
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert (f"{default.name} sampled at the model's default temperature, not the recorded 0.8"
            in text), text
    assert f"{pinned.name} sampled at" not in text


def test_equal_sampling_compares_without_the_label(tmp_path):
    a = _agent_artifact(tmp_path, "a", frozenset({"temperature", "seed"}))
    b = _agent_artifact(tmp_path, "b", frozenset({"temperature", "seed"}))
    text = write_comparison_report(build_comparison([a, b]), reports_dir=tmp_path).read_text()
    assert SAMPLING_MISMATCH_LABEL not in text and "## Difference between models\n" in text


def test_campaign_labels_unequal_sampling(tmp_path):
    default, pinned = _unequal(tmp_path)
    comp = build_comparison([default, pinned], allow_sampling_mismatch=True)
    campaign = Campaign(surfaces=[SurfaceResult("agent", "Agent", comp)],
                        models=["fake:default", "fake:pinned"],
                        requested_models=["fake:default", "fake:pinned"], errors=[])
    section = write_campaign_report(campaign, reports_dir=tmp_path).read_text().split("### Agent")[1]
    assert SAMPLING_MISMATCH_LABEL[1:] in section


def test_anthropic_target_class_default_is_unchanged():
    """The class still declares temperature; only an instance learns otherwise."""
    assert "temperature" in AnthropicTarget.supported_options
