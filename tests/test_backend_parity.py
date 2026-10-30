"""Backend parity: what a backend cannot do is recorded or refused, never silently absorbed.

Four gaps from the backend-capability push gate: `iago compare` ignored a dropped seed; a second
system message became a RUN-ERROR row that dropped out of the rates; the determinism probe recorded
an error without its class; and the agent manifest writers had no way for a `chat_fn` to declare a
control it drops. Each test below names the revert that reds it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iago import agentic_exfil
from iago.adaptive import START, LLMAttacker, NextMove, run_adaptive_conversation
from iago.agent_harness import AgentTrace
from iago.agent_run import run_agent_suite
from iago.artifacts import read_artifact
from iago.campaign import Campaign, SurfaceResult, write_campaign_report
from iago.compare import build_comparison, write_comparison_report
from iago.attacks import Technique
from iago.runner import _determinism_probe, _execute_arm, _execute_planted
from iago.target import Target, UnmappableConversation

from test_adaptive import FORBIDDEN, _ScriptAttacker
from test_delivery_signal import BASE, scripted_chat
from test_target import _anthropic


def _drops_seed(chat_fn, honors=frozenset({"temperature"})):
    """A chat_fn that declares, through the seam, which controls it honors (temperature, not seed)."""
    def fn(messages, tools, options):
        return chat_fn(messages, tools, options)
    fn.supported_options = honors
    return fn


# --- item 4: the capability seam on chat_fn ------------------------------------------------
def test_agent_manifest_records_a_control_the_chat_fn_drops(tmp_path):
    """Revert: drop `unsupported_controls` from agent_run's sampling block -> KeyError here."""
    out = run_agent_suite(_drops_seed(scripted_chat(["done"])), model_name="fake", trials=1,
                          max_steps=2, scenarios=[BASE], artifacts_dir=tmp_path)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == ["seed"]


def test_agent_manifest_records_a_dropped_temperature(tmp_path):
    """Revert: probe only `{"seed": ...}` in the writer -> temperature reads as honored."""
    out = run_agent_suite(_drops_seed(scripted_chat(["done"]), honors=frozenset({"seed"})),
                          model_name="fake", trials=1, max_steps=2, scenarios=[BASE],
                          artifacts_dir=tmp_path)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == ["temperature"]


def test_agent_manifest_records_none_for_a_pass_through_chat_fn(tmp_path):
    out = run_agent_suite(scripted_chat(["done"]), model_name="fake", trials=1, max_steps=2,
                          scenarios=[BASE], artifacts_dir=tmp_path)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == []


def test_exfil_manifest_records_a_control_the_chat_fn_drops(tmp_path):
    """The shared writer behind the five exfil-shaped surfaces. Revert: drop the key -> KeyError."""
    scen = SimpleNamespace(id="s", name="s", kind="attack", owasp=None, asi=None, atlas=None, task="t")
    out = agentic_exfil.run_exfil_suite(
        _drops_seed(scripted_chat(["done"])), model_name="fake", scenarios=[scen],
        run_one=lambda s, chat, options, max_steps: AgentTrace("s", [], "done", 1, False),
        judge=lambda t, s: SimpleNamespace(verdict="resisted", confidence=1.0, rationale=""),
        extra_row=lambda t, s, v: {}, run_id_prefix="parity", judge_modules=("agent_oracle",),
        trials=1, base_seed=0, artifacts_dir=tmp_path)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == ["seed"]


# --- item 1: compare reads the dropped controls ---------------------------------------------
def test_compare_flags_an_artifact_whose_target_dropped_the_seed(tmp_path):
    """Revert: stop filling `Comparison.unsupported` (or drop the warning) -> the line is absent."""
    dropped = run_agent_suite(_drops_seed(scripted_chat(["done"])), model_name="fake:a", trials=1,
                              max_steps=2, scenarios=[BASE], artifacts_dir=tmp_path / "a")
    clean = run_agent_suite(scripted_chat(["done"]), model_name="fake:b", trials=1, max_steps=2,
                            scenarios=[BASE], artifacts_dir=tmp_path / "b")
    comp = build_comparison([dropped, clean])
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    line = next((ln for ln in text.splitlines() if "Sampling controls dropped by the target" in ln),
                "<no dropped-control line in the compare report>")
    assert f"{dropped.name} (seed)" in line, line
    assert clean.name not in line


def test_compare_is_silent_when_no_target_dropped_a_control(tmp_path):
    clean = run_agent_suite(scripted_chat(["done"]), model_name="fake", trials=1, max_steps=2,
                            scenarios=[BASE], artifacts_dir=tmp_path)
    text = write_comparison_report(build_comparison([clean]), reports_dir=tmp_path).read_text()
    assert "Sampling controls dropped" not in text


# --- item 2: a second system message fails the run -----------------------------------------
class _Unmappable(Target):
    name, is_local = "fake:model", True

    def generate(self, prompt, system=None, options=None):
        raise UnmappableConversation("two system messages")

    def chat(self, messages, options=None):
        raise UnmappableConversation("two system messages")


TECH = Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")


def test_anthropic_refuses_a_second_system_message_with_the_run_failing_class():
    target, _ = _anthropic()
    with pytest.raises(UnmappableConversation):
        target.chat([{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                     {"role": "user", "content": "hi"}])


def test_runner_fails_the_run_instead_of_writing_a_run_error_row():
    """Revert: make UnmappableConversation stop subclassing RunFatal -> a <<RUN-ERROR>> reply comes back."""
    with pytest.raises(UnmappableConversation):
        _execute_planted(_Unmappable(), "sys", [], "ask", {})


def test_single_shot_arm_fails_the_run_instead_of_writing_a_run_error_row():
    """`_execute_arm` is the other runner catch site. Revert its `except RunFatal` -> a RUN-ERROR."""
    with pytest.raises(UnmappableConversation):
        _execute_arm(_Unmappable(), TECH, ["ask"], {}, "sys")


def test_adaptive_fails_the_run_instead_of_writing_a_run_error_turn():
    """Revert: drop the adaptive re-raise -> the conversation records a RUN-ERROR turn."""
    with pytest.raises(UnmappableConversation):
        run_adaptive_conversation(_Unmappable(), FORBIDDEN, _ScriptAttacker([NextMove("hi", "ask", "Ask")]), max_turns=1)


# --- item 3: the determinism probe names the exception class --------------------------------
class _Raises(Target):
    name, is_local = "fake:model", True

    def generate(self, prompt, system=None, options=None):
        raise TypeError("bad option")


def test_probe_errors_carry_the_exception_class():
    """Revert: record `str(exc)` again -> the class prefix is gone."""
    rec = _determinism_probe(_Raises(), temperature=0.8, base_seed=1)
    assert rec["warmup"]["error"] == "TypeError: bad option"
    assert [p["error"] for p in rec["probes"]] == ["TypeError: bad option"] * len(rec["probes"])
    assert rec["mismatch_detected"] is None


# --- the same fail-loud rule at every catch site that would otherwise absorb it -------------
def test_trigger_check_fails_instead_of_writing_error_rows(tmp_path):
    """Revert: drop trigger.py's `except RunFatal: raise` -> every arm becomes an error row."""
    from test_trigger import _run
    with pytest.raises(UnmappableConversation):
        _run(_Unmappable(), tmp_path, trials=1)


def test_llm_attacker_does_not_hide_a_run_fatal_error_behind_a_template():
    """Revert: drop the attacker's `except RunFatal: raise` -> a `fallback:` move comes back."""
    def unmappable(msgs, opts):
        raise UnmappableConversation("two system messages")
    with pytest.raises(UnmappableConversation):
        LLMAttacker(1337, FORBIDDEN, chat_fn=unmappable).next_move(0, START, (), [])


def test_campaign_report_carries_the_dropped_control_caveat(tmp_path):
    """Revert: drop the campaign's `dropped_controls_line` call -> the finding loses its caveat."""
    dropped = run_agent_suite(_drops_seed(scripted_chat(["done"])), model_name="fake:a", trials=1,
                              max_steps=2, scenarios=[BASE], artifacts_dir=tmp_path / "a")
    campaign = Campaign(surfaces=[SurfaceResult("agent", "Agent", build_comparison([dropped]))],
                        models=["fake:a"], requested_models=["fake:a"], errors=[])
    text = write_campaign_report(campaign, reports_dir=tmp_path).read_text()
    section = text.split("### Agent")[1]
    assert f"Sampling controls dropped by the target:** {dropped.name} (seed)" in section


def test_ollama_chat_fn_declares_the_controls_it_applies(monkeypatch):
    """Revert: drop the declaration -> the seam reads Ollama as undeclared (pass-through by default)."""
    from iago import agent_run
    from iago.target import unsupported_options
    monkeypatch.setattr(agent_run, "require_ollama_model", lambda model: model)
    fn = agent_run.ollama_chat_fn("llama3.1:latest")
    assert fn.supported_options >= {"temperature", "seed"}
    assert unsupported_options(fn, {"temperature": 0.8, "seed": 1}) == []
