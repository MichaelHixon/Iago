"""--target replay and --record-replay: a recorded run plays back exactly, and never invents a reply."""

import json

import pytest

from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.cli import main
from iago.objectives import Objective
from iago.replay import DEFAULT_REPLAY_FIXTURE, RecordingTarget, ReplayTarget, replay_key
from iago.report import build_html_report, build_report, replay_disclosure
from iago.runner import run
from iago.target import Target, TruncatedReply, build_target

TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden"), Objective("oc", "benign thing", "control")]


class CountingTarget(Target):
    """A different reply on every call, so a replay that collapses repeated calls cannot pass."""

    name = "fake:counting"
    is_local = True

    def __init__(self):
        self.n = 0

    def _next(self):
        self.n += 1
        return f"reply number {self.n} " + "x" * 200

    def generate(self, prompt, system=None, options=None):
        return self._next()

    def chat(self, messages, options=None):
        return self._next()


def test_a_fresh_replay_run_writes_the_fixture_library_hash(tmp_path):
    """The runner records the label-free library fingerprint: the shipped fixture predates the
    ATLAS tags, and a fresh run of the same attack text must still record its hash."""
    header = json.loads(DEFAULT_REPLAY_FIXTURE.read_text().splitlines()[0])
    art = run(ReplayTarget(), trials=3, category=header["category"], artifacts_dir=tmp_path)
    m, _ = read_artifact(art)
    assert m["technique_library_sha256"] == header["technique_library_sha256"]


def _record(tmp_path, trials=2):
    rec = RecordingTarget(CountingTarget())
    art = run(rec, trials=trials, techniques=TECHS, objectives=OBJS, artifacts_dir=tmp_path / "live")
    manifest, rows = read_artifact(art)
    fixture = rec.save(tmp_path / "fx.jsonl", manifest=manifest)
    return fixture, manifest, rows


def _replay(tmp_path, fixture, trials=2):
    art = run(ReplayTarget(fixture), trials=trials, techniques=TECHS, objectives=OBJS,
              artifacts_dir=tmp_path / "replay")
    return read_artifact(art)


def test_replay_reproduces_the_recorded_run_row_for_row(tmp_path):
    fixture, live_m, live_rows = _record(tmp_path)
    m, rows = _replay(tmp_path, fixture)
    pick = lambda rs: [(r["prompt"], r["response"], r["verdict"], r["seed"]) for r in rs]
    assert pick(rows) == pick(live_rows)
    # every trial got its own reply: the seed is part of the key
    assert len({r["response"] for r in rows}) == len(rows)
    # the determinism probe's same-seed pair replays as the recording measured it
    assert m["determinism"]["mismatch_detected"] == live_m["determinism"]["mismatch_detected"] is True


def test_recording_is_transparent_to_the_run(tmp_path):
    rec = RecordingTarget(CountingTarget())
    assert (rec.name, rec.is_local, rec.supported_options) == ("fake:counting", True, None)


def test_a_call_the_fixture_does_not_hold_raises_and_the_runner_records_an_error(tmp_path):
    fixture, _, _ = _record(tmp_path, trials=1)
    t = ReplayTarget(fixture)
    with pytest.raises(RuntimeError, match="replay miss"):
        t.generate("a prompt nobody recorded", options={"seed": 1})
    _, rows = _replay(tmp_path, fixture, trials=2)  # trial 2's seed was never recorded
    assert {r["status"] for r in rows if r["trial"] == 1} == {"error"}
    assert {r["status"] for r in rows if r["trial"] == 0} == {"evaluated"}


def test_replay_names_itself_and_honors_no_sampling_control(tmp_path):
    fixture, _, _ = _record(tmp_path, trials=1)
    m, _ = _replay(tmp_path, fixture, trials=1)
    assert m["model"] == "replay:fake:counting"
    assert m["sampling"]["unsupported_controls"] == []  # both are keyed, so both are honored
    assert m["replay"]["recorded_model"] == "fake:counting"
    assert len(m["replay"]["fixture_sha256"]) == 64
    assert ReplayTarget(fixture).is_local is True


def test_truncated_flag_survives_the_round_trip(tmp_path):
    class Capped(Target):
        name, is_local = "fake:capped", True

        def generate(self, prompt, system=None, options=None):
            return TruncatedReply("cut off " + "y" * 200)

    rec = RecordingTarget(Capped())
    rec.generate("p", options={"seed": 7})
    t = ReplayTarget(rec.save(tmp_path / "c.jsonl"))
    assert isinstance(t.generate("p", options={"seed": 7}), TruncatedReply)


def test_key_separates_seed_and_temperature_and_ignores_the_token_cap():
    msgs = [{"role": "user", "content": "hi"}]
    assert replay_key(msgs, {"seed": 1}) != replay_key(msgs, {"seed": 2})
    assert replay_key(msgs, {"seed": 1, "temperature": 0.1}) != replay_key(msgs, {"seed": 1, "temperature": 0.9})
    assert replay_key(msgs, {"seed": 1, "num_predict": 9}) == replay_key(msgs, {"seed": 1})


def test_replaying_a_backend_that_dropped_the_seed_still_discloses_it(tmp_path):
    class NoSeed(CountingTarget):
        name, supported_options = "fake:noseed", frozenset({"temperature"})

    rec = RecordingTarget(NoSeed())
    art = run(rec, trials=1, techniques=TECHS, objectives=OBJS, artifacts_dir=tmp_path / "live")
    live_m, _ = read_artifact(art)
    m, rows = _replay(tmp_path, rec.save(tmp_path / "fx.jsonl", manifest=live_m), trials=1)
    assert m["sampling"]["unsupported_controls"] == live_m["sampling"]["unsupported_controls"] == ["seed"]
    assert "The target backend ignored `seed`" in build_report(rows, manifest=m)


def test_only_iago_run_offers_the_replay_target():
    from iago.cli import build_parser
    from iago.target import available_targets

    assert "replay" not in available_targets() and "replay" in available_targets(include_run_only=True)
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    offering = sorted(n for n, sp in sub.choices.items()
                      if any("replay" in (a.choices or []) for a in sp._actions if a.dest == "target"))
    assert offering == ["run"]


_HDR = json.dumps({"record": "replay-header", "format": 1}) + "\n"


@pytest.mark.parametrize("body", [
    "", "not json\n", json.dumps({"record": "manifest"}) + "\n", "[1, 2]\n",
    _HDR + '{"replies": []}\n', _HDR + "[1]\n", _HDR + '{"key": "k", "replies": "abc"}\n',
    _HDR + '{"key": "k", "replies": []}\n', _HDR + '{"key": "k", "replies": [{"text": 1}]}\n',
    json.dumps({"record": "replay-header", "format": 2}) + "\n",
])
def test_a_bad_fixture_is_a_clear_value_error(tmp_path, body):
    p = tmp_path / "bad.jsonl"
    p.write_text(body)
    with pytest.raises(ValueError, match="replay"):
        ReplayTarget(p)


def test_registry_builds_replay_and_a_missing_fixture_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="not readable"):
        build_target("replay", model=str(tmp_path / "nope.jsonl"))


def test_record_replay_refuses_to_record_a_replay(capsys):
    assert main(["run", "--target", "replay", "--record-replay", "x.jsonl"]) == 2
    assert "nothing new to record" in capsys.readouterr().err


def test_report_says_replay_in_md_and_html_and_a_live_report_does_not(tmp_path):
    fixture, live_m, live_rows = _record(tmp_path, trials=1)
    m, rows = _replay(tmp_path, fixture, trials=1)
    md, html = build_report(rows, manifest=m), build_html_report(rows, manifest=m)
    for text in (md, html):
        assert "Recorded replay, not a live run" in text
        assert "recorded replay of a local model" in text
        assert "ignored" not in text.split("Recorded replay")[1][:600]  # no dropped-seed alarm
    live_md, live_html = build_report(live_rows, manifest=live_m), build_html_report(live_rows, manifest=live_m)
    assert replay_disclosure(live_m) == ""
    assert "Recorded replay" not in live_md and "Recorded replay" not in live_html
    assert "Target is a local model under the operator's control." in live_md
    assert "the target is a local model under the operator's control." in live_html


def test_a_key_past_its_recorded_replies_is_a_miss_not_a_repeat(tmp_path):
    rec = RecordingTarget(CountingTarget())
    rec.generate("p", options={"seed": 1})
    t = ReplayTarget(rec.save(tmp_path / "once.jsonl"))
    t.generate("p", options={"seed": 1})
    with pytest.raises(RuntimeError, match="replay miss"):
        t.generate("p", options={"seed": 1})


def test_a_chat_call_the_fixture_does_not_hold_raises(tmp_path):
    fixture, _, _ = _record(tmp_path, trials=1)
    with pytest.raises(RuntimeError, match="replay miss"):
        ReplayTarget(fixture).chat([{"role": "user", "content": "unrecorded"}], {"seed": 1})


def test_a_miss_reaches_the_artifact_naming_the_re_record_command(tmp_path):
    fixture, _, _ = _record(tmp_path, trials=1)
    _, rows = _replay(tmp_path, fixture, trials=2)
    assert all("--record-replay" in r["response"] for r in rows if r["status"] == "error")
    assert any(r["status"] == "error" for r in rows)


def test_multi_turn_replay_reproduces_the_recorded_conversation(tmp_path):
    mt = [Technique("mt1", "Crescendo", "multi-turn", "d", "Final ask: {objective}",
                    turns=("prime one", "prime two"))]
    rec = RecordingTarget(CountingTarget())
    live_m, live_rows = read_artifact(run(rec, trials=1, techniques=mt, objectives=OBJS,
                                          artifacts_dir=tmp_path / "live"))
    fx = rec.save(tmp_path / "mt.jsonl", manifest=live_m)
    _, rows = read_artifact(run(ReplayTarget(fx), trials=1, techniques=mt, objectives=OBJS,
                                artifacts_dir=tmp_path / "replay"))
    assert [(r["prompt"], r["response"]) for r in rows] == [(r["prompt"], r["response"]) for r in live_rows]


def test_the_recorder_hands_back_exactly_what_the_target_returned():
    class Capped(Target):
        name, is_local = "fake:capped", True

        def generate(self, prompt, system=None, options=None):
            return TruncatedReply("cut " + "y" * 200)

        def chat(self, messages, options=None):
            return TruncatedReply("cut " + "z" * 200)

    rec = RecordingTarget(Capped())
    assert isinstance(rec.generate("p", options={"seed": 1}), TruncatedReply)
    assert isinstance(rec.chat([{"role": "user", "content": "c"}], {"seed": 1}), TruncatedReply)


def test_a_guarded_replay_still_says_it_is_a_replay(tmp_path):
    from iago.guards import GuardedTarget

    fixture, _, _ = _record(tmp_path, trials=1)
    art = run(GuardedTarget(ReplayTarget(fixture), []), trials=1, techniques=TECHS, objectives=OBJS,
              artifacts_dir=tmp_path / "guarded")
    m, rows = read_artifact(art)
    assert m["replay"]["recorded_model"] == "fake:counting"
    assert "Recorded replay, not a live run" in build_report(rows, manifest=m)


def test_a_replay_of_a_remote_recording_is_not_called_local(tmp_path):
    class Remote(CountingTarget):
        name, is_local = "fake:remote", False

    rec = RecordingTarget(Remote())
    art = run(rec, trials=1, techniques=TECHS, objectives=OBJS, authorized=True,
              artifacts_dir=tmp_path / "live")
    m0, _ = read_artifact(art)
    m, rows = _replay(tmp_path, rec.save(tmp_path / "r.jsonl", manifest=m0), trials=1)
    for text in (build_report(rows, manifest=m), build_html_report(rows, manifest=m)):
        assert "recorded replay of an authorized remote model" in text
        assert "recorded replay of a local model" not in text


def test_registry_default_loads_the_shipped_fixture():
    t = build_target("replay", model=None)
    assert isinstance(t, ReplayTarget) and t.path == DEFAULT_REPLAY_FIXTURE


@pytest.fixture
def outdirs(tmp_path, monkeypatch):
    """Point the CLI's report and artifact directories at tmp, so no test writes into reports/."""
    import iago.cli
    import iago.report
    import iago.runner

    monkeypatch.setattr(iago.runner, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(iago.cli, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(iago.report, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path


def test_cli_record_replay_writes_a_fixture_the_cli_replays_identically(outdirs, monkeypatch):
    import iago.target

    monkeypatch.setitem(iago.target.TARGET_REGISTRY, "fake", lambda model: CountingTarget())
    fx = outdirs / "cli.jsonl"
    flags = ["--trials", "1", "--limit-techniques", "1", "--limit-objectives", "1",
             "--category", "prompt-extraction"]
    assert main(["run", "--target", "fake", "--record-replay", str(fx), *flags]) == 0
    header = json.loads(fx.read_text().splitlines()[0])
    assert header["category"] == "prompt-extraction" and header["iago_version"]
    assert len(header["technique_library_sha256"]) == 64 and header["unsupported_controls"] == []
    live = sorted((outdirs / "artifacts").glob("*.jsonl"))
    assert main(["run", "--target", "replay", "--model", str(fx), *flags]) == 0
    replayed = [p for p in sorted((outdirs / "artifacts").glob("*.jsonl")) if p not in live]
    pick = lambda p: [(r["prompt"], r["response"], r["verdict"]) for r in read_artifact(p)[1]]
    assert pick(replayed[0]) == pick(live[0])


def test_an_unwritable_record_path_is_refused_before_any_model_time(outdirs, monkeypatch, capsys):
    import iago.target

    calls = []
    monkeypatch.setitem(iago.target.TARGET_REGISTRY, "fake", lambda model: calls.append(1) or CountingTarget())
    assert main(["run", "--target", "fake", "--record-replay", str(outdirs), "--smoke"]) == 2
    assert "cannot write" in capsys.readouterr().err and calls == []


def test_a_failed_fixture_write_after_the_run_keeps_the_reports(outdirs, monkeypatch, capsys):
    import iago.target
    from iago.replay import RecordingTarget as RT

    monkeypatch.setitem(iago.target.TARGET_REGISTRY, "fake", lambda model: CountingTarget())
    monkeypatch.setattr(RT, "save", lambda self, path, manifest=None: (_ for _ in ()).throw(OSError("disk full")))
    assert main(["run", "--target", "fake", "--record-replay", str(outdirs / "f.jsonl"), "--smoke",
                 "--html"]) == 2
    assert "could not write the replay fixture" in capsys.readouterr().err
    assert list((outdirs / "reports").glob("*.html")), "the HTML report must be written before the fixture"


def test_cli_records_the_model_reply_not_the_guard_block(outdirs, monkeypatch):
    import iago.cli
    import iago.target
    from iago.guards import Guard, GuardResult

    class BlockAll(Guard):
        name, stage = "block-all", "output"

        def inspect(self, text, *, system=None):
            return GuardResult(True, "test")

    monkeypatch.setitem(iago.target.TARGET_REGISTRY, "fake", lambda model: CountingTarget())
    monkeypatch.setattr(iago.cli, "build_guards", lambda spec: [BlockAll()])
    fx = outdirs / "guarded.jsonl"
    assert main(["run", "--target", "fake", "--guard", "x", "--record-replay", str(fx), "--trials", "1",
                 "--limit-techniques", "1", "--limit-objectives", "1", "--no-determinism-check"]) == 0
    texts = [rep["text"] for line in fx.read_text().splitlines()[1:] for rep in json.loads(line)["replies"]]
    assert texts and all(t.startswith("reply number") for t in texts)


# --- the shipped fixture: a stranger's first run ------------------------------------------------

def _scorecard(md: str) -> list[str]:
    lines = md.splitlines()
    start = lines.index(next(ln for ln in lines if ln.startswith("## Scorecard")))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return lines[start:end]


def test_readme_zero_install_command_runs_offline_and_reproduces_the_sample(outdirs, monkeypatch):
    """The README's command, through the CLI, with Ollama unreachable and no API keys. Its scorecard
    must equal the committed sample's: if the library, judge or fixture drift, re-record
    (samples/README.md)."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert main(["run", "--category", "prompt-extraction", "--target", "replay"]) == 0
    (art,) = (outdirs / "artifacts").glob("*.jsonl")
    m, rows = read_artifact(art)
    assert len(rows) == 108 and not [r for r in rows if r["status"] == "error"], "stale fixture: re-record"
    sample = (DEFAULT_REPLAY_FIXTURE.parent.parent / "prompt-extraction-report.md").read_text()
    assert _scorecard(build_report(rows, manifest=m)) == _scorecard(sample)
    assert "❌ 3 VULNERABLE" in "\n".join(_scorecard(sample))  # the README's stated finding

def test_shipped_fixture_runs_the_smoke_offline_with_a_real_verdict(tmp_path):
    """The README's zero-install command. If this fails after a library or objective edit, the
    shipped fixture is stale: re-record it live (README, "Re-recording the sample")."""
    art = run(ReplayTarget(), trials=1, technique_limit=1, objective_limit=1,
              category="prompt-extraction", artifacts_dir=tmp_path)
    m, rows = read_artifact(art)
    assert [r["status"] for r in rows] == ["evaluated"], "stale fixture: re-record samples/replay/"
    assert rows[0]["verdict"] in {"refused", "bypassed"}  # the canary oracle's fact-grounded verdicts
    assert m["replay"]["fixture"] == DEFAULT_REPLAY_FIXTURE.name


def test_shipped_fixture_replays_its_whole_recorded_run_without_a_miss(tmp_path):
    header = json.loads(DEFAULT_REPLAY_FIXTURE.read_text().splitlines()[0])
    art = run(ReplayTarget(), trials=3, category=header["category"], artifacts_dir=tmp_path)
    _, rows = read_artifact(art)
    assert rows and all(r["status"] == "evaluated" for r in rows), "stale fixture: re-record samples/replay/"
