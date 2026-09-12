"""ISC-49/50 — the determinism probe can fail, says so honestly, and its cost is recorded.

ISC-34 shipped a real check whose probe could not fail: one fixed three-word echo at 24 tokens is a
near-degenerate generation, so the two replies agreed whether or not the host was bit-reproducible.
ISC-50 then fixed what reviewing ISC-49 found: a field named `exact_match` cannot carry "we failed
to disprove", probe 1 straddled the cold-load boundary, and several of these paths were unpinned.
"""

import itertools
import re

import pytest

from iago import runner as runner_mod
from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.objectives import Objective
from iago.runner import (_DETERMINISM_PROBE_TOKENS, _DETERMINISM_PROBES, _DETERMINISM_WARMUP, run)
from iago.target import Target

TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


class _T(Target):
    """Replies by prompt so a probe can be steered independently of the matrix."""

    name, is_local = "fake:model", True

    def __init__(self, replies=None, raise_on=None, raise_after=None, error="transport exploded"):
        self._replies = replies or itertools.repeat("A" * 300)
        self._raise_on = raise_on or ()
        self._raise_after = raise_after or {}     # prompt -> raise once this many calls have run
        self._error = error
        self.calls = []

    def generate(self, prompt, system=None, options=None):
        self.calls.append((prompt, options))
        seen = sum(1 for c in self.calls if c[0] == prompt)
        if prompt in self._raise_on or seen > self._raise_after.get(prompt, 1 << 30):
            raise RuntimeError(self._error)
        return next(self._replies)

    def chat(self, messages, options=None):
        return self.generate(None, options=options)


def _det(target, tmp_path, **kw):
    manifest, _ = read_artifact(run(target, techniques=TECHS, objectives=OBJS, trials=1,
                                    artifacts_dir=tmp_path, **kw))
    return manifest["determinism"]


# ---------------------------------------------------------------- the probes themselves

def test_probes_are_open_ended_not_fixed_echoes():
    """The load-bearing half of ISC-49's claim, pinned as a PROPERTY.

    A test-quality review reintroduced the ISC-34 weakness by swapping in two DISTINCT three-word
    echoes while keeping every other mechanic, and all 685 tests passed: the suite pinned the shape
    of the new probe and never its regime. An echo instruction pins the reply to a point mass, which
    is exactly what made the old probe unable to fail.
    """
    echo_shapes = re.compile(
        r"reply with exactly|respond with exactly|say exactly|repeat (?:the|this|after)"
        r"|output only|the (?:three|four|five|six) words", re.I)
    for text in _DETERMINISM_PROBES:
        assert not echo_shapes.search(text), f"probe is an echo, it cannot fail: {text!r}"
        assert len(text.split()) >= 10, f"probe too short to leave the point-mass regime: {text!r}"
        # Generative verbs: the probe must ask the model to COMPOSE, not to reproduce.
        assert re.search(r"\b(describe|invent|explain|imagine|write|list|tell)\b", text, re.I)


def test_two_distinct_probes_hedge_prompt_dependent_determinism():
    # Not a power argument: this host demonstrated one probe matching while the other differed.
    assert len(_DETERMINISM_PROBES) == 2
    assert _DETERMINISM_PROBES[0] != _DETERMINISM_PROBES[1]
    assert _DETERMINISM_PROBE_TOKENS == 160


# ---------------------------------------------------------------- the happy path

def test_warmup_is_discarded_then_each_probe_fires_twice(tmp_path):
    t = _T()
    d = _det(t, tmp_path, base_seed=777, temperature=0.4)
    assert [c[0] for c in t.calls[:5]] == [
        _DETERMINISM_WARMUP,
        _DETERMINISM_PROBES[0], _DETERMINISM_PROBES[0],
        _DETERMINISM_PROBES[1], _DETERMINISM_PROBES[1]]
    for _, opts in t.calls[:5]:
        assert opts == {"temperature": 0.4, "seed": 777, "num_predict": 160}
    assert d["warmup"] == {"fired": True, "error": None, "prompt": _DETERMINISM_WARMUP}
    # The warm-up is NOT counted as a probe generation, and no probe pair straddles cold load.
    assert d["generations"] == 4
    assert [p["position"] for p in d["probes"]] == ["warm", "warm"]
    assert d["mismatch_detected"] is False
    assert [p["mismatch"] for p in d["probes"]] == [False, False]
    # The recorded options are the artifact's only record of HOW it was measured.
    assert d["options"] == {"temperature": 0.4, "seed": 777, "num_predict": 160}
    assert d["max_tokens_per_generation"] == 160


def test_the_field_never_says_reproducible(tmp_path):
    """One-sidedness in the SCHEMA, not only in prose.

    `exact_match: true` reads as a pass to every downstream consumer. There is no value on this
    field meaning "this host reproduces" because the instrument cannot observe that.
    """
    d = _det(_T(), tmp_path)
    assert "exact_match" not in d
    assert d["mismatch_detected"] is False        # no mismatch OBSERVED, not "reproducible"


def test_sha_pair_records_both_replies_not_one_twice(tmp_path):
    t = _T(replies=iter(["warmup", "first", "second", "x", "x", "A" * 300]))
    d = _det(t, tmp_path)
    a, b = d["probes"][0]["reply_sha256"]
    assert a != b                                  # recording reply A twice is not evidence
    assert d["probes"][1]["reply_sha256"][0] == d["probes"][1]["reply_sha256"][1]


# ---------------------------------------------------------------- failure directions

def test_one_differing_probe_drives_the_aggregate_true_and_names_which(tmp_path):
    t = _T(replies=iter(["warmup", "same", "same", "one", "two", "A" * 300]))
    d = _det(t, tmp_path)
    assert d["mismatch_detected"] is True
    assert [p["mismatch"] for p in d["probes"]] == [False, True]
    assert d["probes"][1]["probe"] == _DETERMINISM_PROBES[1]


def test_both_probes_differing(tmp_path):
    t = _T(replies=iter(["warmup", "a", "b", "c", "d", "A" * 300]))
    d = _det(t, tmp_path)
    assert d["mismatch_detected"] is True
    assert [p["mismatch"] for p in d["probes"]] == [True, True]


def test_a_raising_probe_gives_null_never_false(tmp_path):
    t = _T(raise_on=(_DETERMINISM_PROBES[1],))
    d = _det(t, tmp_path)
    assert d["mismatch_detected"] is None          # unknown is not "no mismatch"
    assert [p["mismatch"] for p in d["probes"]] == [False, None]
    assert "transport exploded" in d["probes"][1]["error"]
    assert d["generations"] == 2


def test_raise_on_the_SECOND_call_reconciles_generations_with_hashes(tmp_path):
    """The odd-generations path. A partial pair records the one hash it got."""
    t = _T(raise_after={_DETERMINISM_PROBES[0]: 1})
    d = _det(t, tmp_path)
    p0 = d["probes"][0]
    assert p0["mismatch"] is None and p0["replies_returned"] == 1
    assert len(p0["reply_sha256"]) == 1            # reconciles with the count, does not vanish
    assert d["generations"] == 3                   # 1 + 2, and the hashes across probes total 3
    assert sum(len(p["reply_sha256"]) for p in d["probes"]) == 3
    assert d["mismatch_detected"] is None


def test_both_probes_raising(tmp_path):
    t = _T(raise_on=tuple(_DETERMINISM_PROBES))
    d = _det(t, tmp_path)
    assert d["mismatch_detected"] is None
    assert d["generations"] == 0
    assert all(p["mismatch"] is None for p in d["probes"])


def test_a_mismatch_outranks_a_raise(tmp_path):
    t = _T(replies=iter(["warmup", "one", "two", "A" * 300]), raise_on=(_DETERMINISM_PROBES[1],))
    d = _det(t, tmp_path)
    assert d["mismatch_detected"] is True          # a raise cannot un-observe a difference


def test_a_mismatch_outranks_a_raise_in_EITHER_order(tmp_path):
    """Precedence must be a rule over the whole set, not whichever probe reported first.

    An order-sensitive aggregate (first non-True wins) passes every other test here, because the
    only raise-plus-mismatch fixture happens to put the mismatch first.
    """
    t = _T(replies=iter(["warmup", "one", "two"]), raise_on=(_DETERMINISM_PROBES[0],))
    d = _det(t, tmp_path)
    assert [p["mismatch"] for p in d["probes"]] == [None, True]
    assert d["mismatch_detected"] is True


def test_an_empty_probe_set_is_unknown_never_clean(tmp_path, monkeypatch):
    """The guard that keeps a zero-probe config from manufacturing a pass."""
    monkeypatch.setattr(runner_mod, "_DETERMINISM_PROBES", ())
    d = _det(_T(), tmp_path)
    assert d["mismatch_detected"] is None
    assert d["probes"] == [] and d["generations"] == 0


def test_a_failed_warmup_marks_probe_one_cold(tmp_path):
    """If the warm-up did not run, probe 1 still straddles the cold boundary — say so."""
    t = _T(raise_on=(_DETERMINISM_WARMUP,))
    d = _det(t, tmp_path)
    assert d["warmup"]["fired"] is False and "transport exploded" in d["warmup"]["error"]
    assert [p["position"] for p in d["probes"]] == ["cold", "warm"]


def test_error_text_is_truncated_not_dumped(tmp_path):
    t = _T(raise_on=(_DETERMINISM_PROBES[0],), error="E" * 5000)
    d = _det(t, tmp_path)
    assert len(d["probes"][0]["error"]) == 200


# ---------------------------------------------------------------- operator-facing surface

def test_the_warning_names_which_probe_differed(tmp_path, capsys):
    t = _T(replies=iter(["warmup", "same", "same", "one", "two", "A" * 300]))
    run(t, techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path, progress=True)
    out = capsys.readouterr().out
    assert "probe 2 of 2 produced two DIFFERENT replies" in out


def test_no_warning_when_nothing_differed(tmp_path, capsys):
    run(_T(), techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path, progress=True)
    assert "DIFFERENT replies" not in capsys.readouterr().out


# ---------------------------------------------------------------- the skip flag

def test_skipping_records_null_and_fires_zero_generations(tmp_path):
    t = _T()
    manifest, rows = read_artifact(run(t, techniques=TECHS, objectives=OBJS, trials=1,
                                       artifacts_dir=tmp_path, determinism_check=False))
    assert manifest["determinism"] is None
    assert len(t.calls) == 1                       # the single trial, no warm-up, no probes
    assert len(rows) == 1


def test_probe_replies_are_never_trial_rows(tmp_path):
    _, rows = read_artifact(run(_T(), techniques=TECHS, objectives=OBJS, trials=2,
                                artifacts_dir=tmp_path))
    assert len(rows) == 2
    prompts = {r.get("prompt") for r in rows}
    assert not (prompts & (set(_DETERMINISM_PROBES) | {_DETERMINISM_WARMUP}))
