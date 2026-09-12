"""ISC-49 — the determinism probe can fail, and its cost is recorded rather than promised.

ISC-34 shipped a real check whose probe could not fail: one fixed three-word echo at 24 tokens is a
near-degenerate generation, so the two replies agreed whether or not the host was bit-reproducible.
These tests pin the strengthened shape, and the one-sidedness that survives it.
"""

import itertools

from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.objectives import Objective
from iago.runner import _DETERMINISM_PROBES, _DETERMINISM_PROBE_TOKENS, run
from iago.target import Target

TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


class _T(Target):
    name, is_local = "fake:model", True

    def __init__(self, replies=None, raise_on=None):
        self._replies = replies or itertools.repeat("A" * 300)
        self._raise_on = raise_on
        self.calls = []

    def generate(self, prompt, system=None, options=None):
        self.calls.append((prompt, options))
        if self._raise_on is not None and prompt == self._raise_on:
            raise RuntimeError("transport exploded")
        return next(self._replies)

    def chat(self, messages, options=None):
        return self.generate(None, options=options)


def _manifest(target, tmp_path, **kw):
    manifest, _ = read_artifact(run(target, techniques=TECHS, objectives=OBJS, trials=1,
                                    artifacts_dir=tmp_path, **kw))
    return manifest["determinism"]


def test_probes_are_distinct_open_ended_and_fired_twice_each(tmp_path):
    t = _T()
    d = _manifest(t, tmp_path, base_seed=777, temperature=0.4)
    probe_calls = t.calls[:4]
    # Each probe fired exactly twice, in order, at the run's temperature and base seed.
    assert [c[0] for c in probe_calls] == [_DETERMINISM_PROBES[0], _DETERMINISM_PROBES[0],
                                           _DETERMINISM_PROBES[1], _DETERMINISM_PROBES[1]]
    assert len(set(_DETERMINISM_PROBES)) == 2      # two DISTINCT probes, not one fired four times
    for _, opts in probe_calls:
        assert opts == {"temperature": 0.4, "seed": 777, "num_predict": 160}
    # Long enough to leave the point-mass regime the 24-token echo sat in.
    assert _DETERMINISM_PROBE_TOKENS >= 100
    assert d["generations"] == 4 and len(d["probes"]) == 2
    assert d["exact_match"] is True
    assert all(p["exact_match"] is True and len(p["reply_sha256"]) == 2 for p in d["probes"])


def test_one_differing_probe_drives_the_aggregate_false(tmp_path):
    # Probe 1 matches, probe 2 differs: the aggregate is False, and the manifest says WHICH.
    t = _T(replies=iter(["same", "same", "one", "two", "A" * 300]))
    d = _manifest(t, tmp_path)
    assert d["exact_match"] is False
    assert [p["exact_match"] for p in d["probes"]] == [True, False]
    assert d["probes"][1]["probe"] == _DETERMINISM_PROBES[1]
    assert d["generations"] == 4


def test_a_raising_probe_drives_the_aggregate_null_never_false(tmp_path):
    # A probe that could not run is NOT a mismatch. null is neither a pass nor a failure.
    t = _T(replies=itertools.repeat("A" * 300), raise_on=_DETERMINISM_PROBES[1])
    d = _manifest(t, tmp_path)
    assert d["exact_match"] is None
    assert [p["exact_match"] for p in d["probes"]] == [True, None]
    assert "transport exploded" in d["probes"][1]["error"]
    assert d["generations"] == 2            # only the returned calls are counted


def test_a_mismatch_outranks_a_raise(tmp_path):
    # False is definitive: a probe that raised cannot un-observe a probe that differed.
    t = _T(replies=iter(["one", "two", "A" * 300]), raise_on=_DETERMINISM_PROBES[1])
    d = _manifest(t, tmp_path)
    assert d["exact_match"] is False


def test_skipping_records_null_and_fires_zero_generations(tmp_path):
    t = _T()
    manifest, rows = read_artifact(run(t, techniques=TECHS, objectives=OBJS, trials=1,
                                       artifacts_dir=tmp_path, determinism_check=False))
    assert manifest["determinism"] is None
    assert len(t.calls) == 1                # the single trial, and nothing else
    assert len(rows) == 1


def test_probe_replies_are_never_trial_rows(tmp_path):
    t = _T()
    path = run(t, techniques=TECHS, objectives=OBJS, trials=2, artifacts_dir=tmp_path)
    _, rows = read_artifact(path)
    assert len(rows) == 2
    prompts = {r.get("prompt") for r in rows}
    assert not (prompts & set(_DETERMINISM_PROBES))
