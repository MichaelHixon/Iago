"""`--category`: restrict a run to one row of `iago library`, applied before the positional limits.

The filter drops objectives no filtered technique fires at, so `--smoke --category prompt-extraction`
picks the first COMPATIBLE objective instead of `obj-phish` and raising (Council finding, 2026-09-23).
"""

import types

import pytest

from iago import cli
from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.config import CATEGORIES
from iago.guards_thirdparty import GuardBackendUnavailable
from iago.objectives import Objective
from iago.runner import load_artifacts, run
from iago.target import Target


class _T(Target):
    name, is_local = "fake:model", True

    def generate(self, prompt, system=None, options=None):
        return "A" * 300

    def chat(self, messages, options=None):
        return self.generate(None)


LIB = [
    Technique("d1", "Direct one", "direct-ask", "d", "Do: {objective}"),
    Technique("r1", "Persona one", "role-play", "d", "Act: {objective}"),
    Technique("r2", "Persona two", "role-play", "d", "Play: {objective}"),
    Technique("d2", "Direct two", "direct-ask", "d", "Now: {objective}"),
]
FORBIDDEN = Objective("o1", "forbidden thing", "forbidden")
CONTROL = Objective("oc", "benign thing", "control")


def _kw(tmp_path, **kw):
    base = dict(trials=1, artifacts_dir=tmp_path, determinism_check=False, objectives=[FORBIDDEN])
    base.update(kw)
    return base


def test_category_keeps_only_that_category(tmp_path):
    rows = load_artifacts(run(_T(), techniques=LIB, category="role-play", **_kw(tmp_path)))
    assert sorted(r["technique_id"] for r in rows) == ["r1", "r2"]


def test_category_filters_before_the_positional_limit(tmp_path):
    # The library's first technique is d1; the first ROLE-PLAY technique is r1.
    rows = load_artifacts(run(_T(), techniques=LIB, category="role-play", technique_limit=1,
                              **_kw(tmp_path)))
    assert [r["technique_id"] for r in rows] == ["r1"]


def test_category_drops_incompatible_objectives_before_the_objective_limit(tmp_path):
    # Without the objective pre-filter, objective_limit=1 would take CONTROL, the technique would
    # not fire at it, and the run would raise "no technique/objective pairs fire" (--smoke case).
    forbidden_only = [Technique("f1", "Scoped", "direct-ask", "d", "Do: {objective}",
                                applies_to=("forbidden",))]
    rows = load_artifacts(run(_T(), techniques=forbidden_only, category="direct-ask",
                              objective_limit=1, **_kw(tmp_path, objectives=[CONTROL, FORBIDDEN])))
    assert [r["objective_id"] for r in rows] == ["o1"]


def test_unknown_category_raises_before_writing(tmp_path):
    with pytest.raises(ValueError, match="unknown category 'nope'"):
        run(_T(), techniques=LIB, category="nope", **_kw(tmp_path))
    assert list(tmp_path.glob("*.jsonl")) == []


def test_category_with_no_techniques_raises_no_techniques_to_run(tmp_path):
    with pytest.raises(ValueError, match="no techniques to run"):
        run(_T(), techniques=LIB, category="many-shot", **_kw(tmp_path))


def test_zero_fireable_pairs_hint_names_the_flag(tmp_path):
    leak_only = [Technique("pe-x", "Extract", "prompt-extraction", "d", "Leak: {objective}",
                           applies_to=("prompt-leak",))]
    with pytest.raises(ValueError, match=r"--category"):
        run(_T(), techniques=leak_only, **_kw(tmp_path))


def test_manifest_records_the_category(tmp_path):
    manifest, _ = read_artifact(run(_T(), techniques=LIB, category="role-play", **_kw(tmp_path)))
    assert manifest["category"] == "role-play"
    manifest, _ = read_artifact(run(_T(), techniques=LIB, **_kw(tmp_path)))
    assert manifest["category"] is None


@pytest.mark.parametrize("cmd", ["run", "defense-delta"])
def test_parser_accepts_a_known_category(cmd):
    args = cli.build_parser().parse_args([cmd, "--category", "prompt-extraction"])
    assert args.category == "prompt-extraction"


@pytest.mark.parametrize("cmd", ["run", "defense-delta"])
def test_parser_rejects_an_unknown_category(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([cmd, "--category", "nope"])
    # Only the exit code is pinned: "invalid choice" is CPython's argparse wording, not Iago's,
    # and a Python upgrade that rewords it must not red a test about the category flag.
    assert exc.value.code == 2


def _dd_args(**kw):
    base = dict(target="ollama", model="llama3.1", guard="all", trials=1, temperature=0.8,
                base_seed=1, limit_techniques=None, limit_objectives=None, category=None,
                shots=None, smoke=True, authorized=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize("exc", [ValueError("unknown category 'x'"),
                                 GuardBackendUnavailable("backend down")])
def test_defense_delta_exits_2_on_runner_errors(monkeypatch, capsys, exc):
    # Both were uncaught tracebacks in _cmd_defense_delta before this change.
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(cli, "run", boom)
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "build_guards", lambda spec: [types.SimpleNamespace(name="g")])
    monkeypatch.setattr(cli, "GuardedTarget", lambda base, guards: base)
    rc = cli._cmd_defense_delta(_dd_args())
    assert rc == 2
    err = capsys.readouterr().err
    # Distinctive per arm: a bare "ERROR:" check cannot tell which except clause fired, and would
    # pass silently if GuardBackendUnavailable ever stopped subclassing RuntimeError.
    expected = ("guard backend failed mid-run" if isinstance(exc, GuardBackendUnavailable)
                else "unknown category")
    assert expected in err


# --- the CLI must actually FORWARD the flag ------------------------------------------------
# Deleting `category=...` from either handler reds nothing without these: the other handler
# tests take `**kwargs` and discard them, so the flag could be a no-op in the one place a user
# touches it (test-analyzer finding, 2026-09-23).

def test_cmd_run_forwards_the_category_to_the_runner(tmp_path, monkeypatch):
    seen = {}

    def fake_run(target, **kwargs):
        seen.update(kwargs)
        from iago.runner import run as real_run
        return real_run(_T(), techniques=LIB, objectives=[FORBIDDEN], trials=1,
                        artifacts_dir=tmp_path, determinism_check=False)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "write_report", lambda rows, **kw: tmp_path / "r.md")
    from tests.test_cli_exit_paths import _run_args
    cli._cmd_run(_run_args(category="role-play"))
    assert seen["category"] == "role-play"


def test_cmd_defense_delta_forwards_the_category_to_both_arms(tmp_path, monkeypatch):
    seen = []

    def fake_run(target, **kwargs):
        seen.append(kwargs.get("category"))
        from iago.runner import run as real_run
        return real_run(_T(), techniques=LIB, objectives=[FORBIDDEN], trials=1,
                        artifacts_dir=tmp_path, determinism_check=False)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "build_guards", lambda spec: [types.SimpleNamespace(name="g")])
    monkeypatch.setattr(cli, "GuardedTarget", lambda base, guards: base)
    monkeypatch.setattr(cli, "write_delta_report", lambda *a, **k: tmp_path / "d.md", raising=False)
    cli._cmd_defense_delta(_dd_args(category="role-play"))
    assert seen == ["role-play", "role-play"]  # raw arm AND guarded arm


# --- the objective pre-filter keeps an objective ANY member fires at, not ALL ----------------

def test_objective_prefilter_keeps_objectives_any_member_fires_at(tmp_path):
    # Mixed scoping inside one category: `any` keeps both objectives, `all` would drop both and
    # raise "no objectives to run". A single-technique fixture cannot tell the two apart.
    mixed = [
        Technique("m1", "Forbidden-only", "role-play", "d", "A: {objective}",
                  applies_to=("forbidden",)),
        Technique("m2", "Control-only", "role-play", "d", "B: {objective}",
                  applies_to=("control",)),
    ]
    rows = load_artifacts(run(_T(), techniques=mixed, category="role-play",
                              **_kw(tmp_path, objectives=[FORBIDDEN, CONTROL])))
    assert sorted({r["objective_id"] for r in rows}) == ["o1", "oc"]


# --- --smoke --category must fire for EVERY shipped category --------------------------------

@pytest.mark.parametrize("category", CATEGORIES)
def test_smoke_plus_category_fires_for_every_shipped_category(category, monkeypatch, tmp_path):
    # Drives the real run() against the real library with --smoke's limits, so it pins the
    # BEHAVIOR rather than re-implementing the selection logic in the test. Before the objective
    # filter moved below the technique slice this held only by library ordering.
    rows = load_artifacts(run(_T(), category=category, technique_limit=1, objective_limit=1,
                              trials=1, artifacts_dir=tmp_path, determinism_check=False))
    assert rows, f"--smoke --category {category} produced no trials"


def test_smoke_pairing_survives_a_narrower_sibling_technique(tmp_path):
    # The latent break the reviewer named: the first technique of a category is scoped to one
    # objective kind while a sibling is scoped to another. Filtering objectives against the
    # UNSLICED library would keep the sibling's objective, objs[:1] would take it, and the run
    # would raise the error the filter exists to prevent.
    siblings = [
        Technique("s1", "Forbidden-only", "role-play", "d", "A: {objective}",
                  applies_to=("forbidden",)),
        Technique("s2", "Control-only", "role-play", "d", "B: {objective}",
                  applies_to=("control",)),
    ]
    rows = load_artifacts(run(_T(), techniques=siblings, category="role-play", technique_limit=1,
                              objective_limit=1,
                              **_kw(tmp_path, objectives=[CONTROL, FORBIDDEN])))
    assert [(r["technique_id"], r["objective_id"]) for r in rows] == [("s1", "o1")]


def test_report_discloses_a_partial_run(tmp_path):
    from iago.report import scope_disclosure
    manifest, _ = read_artifact(run(_T(), techniques=LIB, category="role-play", **_kw(tmp_path)))
    text = scope_disclosure(manifest)
    assert "Partial run" in text and "role-play" in text
    # A full run must say nothing, or every report grows a meaningless banner.
    full, _ = read_artifact(run(_T(), techniques=LIB, **_kw(tmp_path)))
    assert scope_disclosure(full) == ""


def test_defense_delta_exits_2_when_the_guard_BACKEND_is_missing(monkeypatch, capsys):
    # Distinct from the mid-run case above: `build_guards` fires a live probe at BUILD time
    # (guards_thirdparty.build_thirdparty_guard), and that raise happens before the try block
    # around run(). It escaped as a traceback until the build-time except tuple was widened.
    def missing(spec):
        raise GuardBackendUnavailable("llama-guard backend not installed")

    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "build_guards", missing)
    rc = cli._cmd_defense_delta(_dd_args(guard="llama-guard"))
    assert rc == 2
    assert "llama-guard backend not installed" in capsys.readouterr().err
