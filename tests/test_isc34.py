"""ISC-34 — the reproducibility claim matches what is recorded and measured."""

import itertools
import re
from pathlib import Path

from iago.agent_harness import LLMMessage
from iago.agent_run import run_agent_suite
from iago.agent_scenarios import Scenario
from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.objectives import Objective
from iago.runner import run
from iago.target import Target

TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


class _T(Target):
    name, is_local = "fake:model", True

    def __init__(self, replies=None):
        self._replies = replies or itertools.repeat("A" * 300)
        self.calls = []

    def generate(self, prompt, system=None, options=None):
        self.calls.append((prompt, options))
        return next(self._replies)

    def chat(self, messages, options=None):
        return self.generate(None, options=options)


def test_deterministic_target_records_no_mismatch_and_probe_is_not_a_row(tmp_path):
    t = _T()
    path = run(t, techniques=TECHS, objectives=OBJS, trials=2, base_seed=777, artifacts_dir=tmp_path)
    manifest, rows = read_artifact(path)
    d = manifest["determinism"]
    # A literal, not the value the code also wrote to the other field it is compared against.
    # ISC-50 renamed the field: a one-sided check cannot report `exact_match: true`.
    assert d["mismatch_detected"] is False and d["options"]["seed"] == 777
    assert manifest["sampling"]["base_seed"] == 777
    assert len(rows) == 2                       # 1 pair x 2 trials; the probe calls are not rows
    assert len(t.calls) == 7                    # 1 warm-up + 4 probe + 2 trials — ISC-49/50
    assert t.calls[0][1]["num_predict"] == 160  # capped, but past the point-mass regime


def test_nondeterministic_target_records_mismatch(tmp_path):
    t = _T(replies=iter(["warmup", "one", "two", "same", "same", "A" * 300]))
    manifest, _ = read_artifact(run(t, techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path))
    assert manifest["determinism"]["mismatch_detected"] is True


def test_check_can_be_skipped_and_records_null(tmp_path):
    manifest, _ = read_artifact(run(_T(), techniques=TECHS, objectives=OBJS, trials=1,
                                    artifacts_dir=tmp_path, determinism_check=False))
    assert manifest["determinism"] is None


def test_agent_path_pins_seed_per_trial(tmp_path):
    seen = []

    def chat_fn(messages, tools, options):
        seen.append(dict(options))
        return LLMMessage(content="done")

    scen = Scenario(id="s", name="s", kind="attack", task="t", documents={"d": "x"},
                    private_notes="secret: C-1", canary="C-1", owasp="LLM01")
    run_agent_suite(chat_fn, model_name="m", trials=3, base_seed=40, temperature=0.3,
                    scenarios=[scen], artifacts_dir=tmp_path)
    assert [o["seed"] for o in seen] == [40, 41, 42]
    assert {o["temperature"] for o in seen} == {0.3}


_ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_doc_states_the_bounded_claim():
    # The full claim lives in docs/; the README carries a one-line pointer to it (2026-09-23).
    # Short anchors, not sentences or markup: the wording is free to change, the claims are not.
    text = _ROOT.joinpath("docs", "reproducibility.md").read_text()
    assert "# Reproducibility" in text
    assert re.search(r"Ollama build, model digest,? and parallelism setting", text)
    assert "reproducible everywhere" in text
    # ISC-50: the probe may not be presented as a warrant that the run DOES reproduce.
    assert re.search(r"\bdisprove determinism\b", text)
    assert re.search(r"\bone-sided\b", text)
    assert re.search(r"this host reproduces", text)
    # The negative half: the field ISC-50 renamed away must never reappear, and the doc must
    # name the one-sided field that replaced it.
    assert "exact_match" not in text
    assert "mismatch_detected" in text


def test_readme_keeps_the_credibility_anchors():
    # Short anchors, not sentences: the wording is free to change, the claims are not.
    text = _ROOT.joinpath("README.md").read_text()
    for anchor in ("--authorized", "95% CI", "planted secret", "three arms", "docs/reproducibility.md"):
        assert anchor in text, anchor
    assert re.search(r"\bdirectional\b", text)            # not satisfied by "bidirectional"
    # Every docs/ link on the page must point at a file that ships with the repo.
    links = re.findall(r"\]\((?:\./)?(docs/[\w-]+\.md)(?:\s+\"[^\"]*\")?\)", text)
    assert links, "README lost every docs/ link"
    for rel in set(links):
        assert _ROOT.joinpath(rel).is_file(), rel
    # ISC-50 on the front page: the probe is framed as a disproof, never as a pass.
    assert re.search(r"\bdisprove\b", text)
    assert "exact_match" not in text
