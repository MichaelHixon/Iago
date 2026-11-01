"""The Batch sweep path (ISC-148..150): `batch-submit` renders the runner's own plan, bounded by a
budget; `batch-collect` reconciles results by custom_id into the runner's artifact shape.

Batch results are built with the installed SDK's own models
(`anthropic.types.messages.MessageBatchIndividualResponse.model_validate`), so every fixture is a
shape the SDK accepts from the wire; the refusal and error bodies follow the documented shapes the
temperature plan's § Premises captured (the 400 body) and the docs give (stop_details).
"""

from __future__ import annotations

import copy
import json
import random
import re
from types import SimpleNamespace

import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from iago import batch as b
from iago import cli
from iago.artifacts import read_artifact
from iago.attacks import load_library
from iago.compare import build_comparison
from iago.objectives import load_objectives
from iago.regrade import regrade_file
from iago.report import batch_disclosure, build_report, verdict_of
from iago.runner import _fires, run
from iago.target import AnthropicTarget, SamplingRefused

MODEL = "claude-haiku-5-5"
NO_TEMP = ["num_predict"]               # what the 2026-11-01 probe left on the 5.5 models
WITH_TEMP = ["num_predict", "temperature"]
DEPRECATED = "`temperature` is deprecated for this model."  # the captured 400 message
SMALL_CLI = ["--limit-techniques", "3", "--trials", "1"]


@pytest.fixture(scope="module")
def lib():
    return load_library()


@pytest.fixture(scope="module")
def objs():
    return load_objectives()


@pytest.fixture(scope="module")
def small(lib, objs):
    """Two single-shot techniques: one plain (fires at forbidden objectives), one provenance
    technique (fires at trust-escalation objectives, three arms each)."""
    plain = next(t for t in lib if not t.is_multiturn and not t.provenance
                 and any(_fires(t, o) and o.kind == "forbidden" for o in objs))
    prov = next(t for t in lib if not t.is_multiturn and t.provenance
                and any(_fires(t, o) and o.is_trust_escalation for o in objs))
    return [plain, prov]


def _sampling(trials=2, technique_limit=None):
    return {"trials": trials, "temperature": 0.8, "base_seed": 1337, "category": None,
            "technique_limit": technique_limit, "objective_limit": None, "shots": None}


def _plan(supported=NO_TEMP, *, techniques=None, objectives=None, trials=2, model=MODEL,
          technique_limit=None):
    return b.plan_requests(model, supported, probe="test", sampling=_sampling(trials, technique_limit),
                           techniques=techniques, objectives=objectives)


@pytest.fixture(scope="module")
def full_plan():
    """The whole library at 2 trials, planned once for the module."""
    return _plan(trials=2)


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _client(**namespaces):
    """A fake Anthropic client: `_client(messages=obj)`, or batches via `_client(batches=obj)`."""
    if "batches" in namespaces:
        return SimpleNamespace(messages=SimpleNamespace(batches=namespaces["batches"]))
    return SimpleNamespace(**namespaces)


class _Answering:
    """`client.messages` for a synchronous run: records each call, replies via `reply(kwargs)`."""

    def __init__(self, reply=lambda kw: "x") -> None:
        self.calls, self.reply = [], reply

    def create(self, **kw):
        self.calls.append(kw)
        return {"content": [{"type": "text", "text": self.reply(kw)}], "stop_reason": "end_turn",
                "stop_details": None, "model": MODEL}


def _ok(cid, text=None, model=MODEL, **msg):
    return MessageBatchIndividualResponse.model_validate({
        "custom_id": cid,
        "result": {"type": "succeeded", "message": {
            "id": f"msg_{cid}", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text if text is not None else f"reply for {cid}"}],
            "stop_reason": "end_turn", "stop_sequence": None, "stop_details": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}, **msg}}})


def _all_ok(plan):
    return [_ok(r["custom_id"]) for r in plan["requests"]]


def _errored(cid, etype="api_error", message="Internal server error"):
    return MessageBatchIndividualResponse.model_validate({
        "custom_id": cid, "result": {"type": "errored", "error": {
            "type": "error", "error": {"type": etype, "message": message}}}})


def _expired(cid):
    return MessageBatchIndividualResponse.model_validate(
        {"custom_id": cid, "result": {"type": "expired"}})


def _results(items):
    return {i.custom_id: i.result for i in items}


# --- ISC-148: the plan ---------------------------------------------------------------------------

def test_submit_writes_stable_ids(lib, objs, full_plan):
    """Two renders give the same requests, one per single-shot call the runner fires (three per
    trust-escalation pair), every id legal for the Batches API."""
    ids = [r["custom_id"] for r in full_plan["requests"]]
    assert _plan(trials=2)["requests"] == full_plan["requests"]
    calls_per_trial = sum(3 if o.is_trust_escalation else 1
                          for t in lib if not t.is_multiturn for o in objs if _fires(t, o))
    assert len(ids) == len(set(ids)) == calls_per_trial * 2
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids)
    assert {i.rsplit("_", 1)[1] for i in ids} == {"1", "anon", "plac", "prov"}


def test_submit_omits_dropped_temperature(small, objs):
    dropped = _plan(NO_TEMP, techniques=small, objectives=objs)
    assert dropped["sampling"]["unsupported_controls"] == ["seed", "temperature"]
    assert not any("temperature" in r["params"] for r in dropped["requests"])
    kept = _plan(WITH_TEMP, techniques=small, objectives=objs)
    assert kept["sampling"]["unsupported_controls"] == ["seed"]
    assert all(r["params"]["temperature"] == 0.8 for r in kept["requests"])


def test_submit_never_sends_fallbacks(full_plan):
    assert all(set(r["params"]) <= {"model", "max_tokens", "system", "messages", "temperature"}
               for r in full_plan["requests"])


def test_submit_lists_multiturn_techniques_as_excluded(lib, full_plan):
    multiturn = sorted(t.id for t in lib if t.is_multiturn)
    assert multiturn and full_plan["excluded_multiturn"] == multiturn
    assert not any(r["custom_id"].split("_")[0] in multiturn for r in full_plan["requests"])


def test_plan_matches_the_runners_requests(small, objs, tmp_path):
    """The planned params are what a synchronous run sends: the same system prompt per objective,
    the same user turn, the same max_tokens (checked against an independent recording)."""
    plan = _plan(techniques=small, objectives=objs)
    messages = _Answering()
    run(AnthropicTarget(MODEL, client=_client(messages=messages), supported_options=NO_TEMP),
        authorized=True, techniques=small, objectives=objs, trials=2, temperature=0.8,
        base_seed=1337, artifacts_dir=tmp_path, determinism_check=False)
    assert [r["params"] for r in plan["requests"]] == messages.calls


# --- ISC-150: the budget ------------------------------------------------------------------------

def test_ceiling_is_a_bound(small, objs):
    plan = _plan(techniques=small, objectives=objs)
    est = plan["estimate"]
    assert est["ceiling"]["output_tokens"] == sum(r["params"]["max_tokens"] for r in plan["requests"])
    assert est["ceiling"]["input_tokens"] >= est["expected"]["input_tokens"]
    assert est["ceiling"]["dollars"] >= est["expected"]["dollars"] > 0


def test_unknown_model_has_no_estimate(small, objs):
    with pytest.raises(ValueError, match="no Batch price"):
        _plan(techniques=small, objectives=objs, model="claude-unpriced-9")


def test_submit_refuses_over_budget(tmp_path, capsys):
    out = tmp_path / "plan.json"
    code = cli.main(["batch-submit", "--model", MODEL, "--model", "claude-opus-5-5",
                     "--max-dollars", "0.0001", "--dry-run", "--out", str(out), *SMALL_CLI])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()


def test_submit_prints_the_estimate(tmp_path, capsys):
    out = tmp_path / "plan.json"
    code = cli.main(["batch-submit", "--model", MODEL, "--max-dollars", "5", "--dry-run",
                     "--out", str(out), *SMALL_CLI])
    stdout = capsys.readouterr().out
    assert code == 0
    assert re.search(rf"{MODEL}: \d+ requests; expected .* = \$\d+\.\d\d; ceiling .* = \$\d+\.\d\d", stdout)
    assert "total at Batch prices" in stdout and "8 multi-turn technique(s) excluded" in stdout
    (plan,) = b.read_record(out)
    assert "batch_id" not in plan and not any("temperature" in r["params"] for r in plan["requests"])


def test_live_submit_requires_authorized(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_batch_client", lambda **kw:pytest.fail("no client before --authorized"))
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: pytest.fail("no probe before --authorized"))
    assert cli.main(["batch-submit", "--model", MODEL, "--max-dollars", "5"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_live_submit_sends_the_plan_and_records_the_batch_id(tmp_path, monkeypatch):
    created = []

    class Batches:
        def create(self, requests):
            created.append(requests)
            return SimpleNamespace(id=f"msgbatch_{len(created)}")

    monkeypatch.setattr(cli, "_batch_client", lambda **kw:_client(batches=Batches()))
    monkeypatch.setattr(cli, "build_target",
                        lambda kind, model: SimpleNamespace(supported_options=frozenset(NO_TEMP)))
    out = tmp_path / "plan.json"
    assert cli.main(["batch-submit", "--model", MODEL, "--max-dollars", "5", "--authorized",
                     "--out", str(out), *SMALL_CLI]) == 0
    (plan,) = b.read_record(out)
    assert plan["batch_id"] == "msgbatch_1" and plan["probe"].startswith("build-time probe")
    assert created[0] == [{"custom_id": r["custom_id"], "params": r["params"]} for r in plan["requests"]]


# --- ISC-149: collection -------------------------------------------------------------------------

def _collect(plan, items, small, objs, tmp_path):
    path = b.collect_plan(plan, _results(items), techniques=small, objectives=objs,
                          artifacts_dir=tmp_path)
    return read_artifact(path)


def test_collect_keys_results_by_custom_id(small, objs, tmp_path):
    """Results arrive shuffled; each row still gets the reply submitted under its own id."""
    plan = _plan(techniques=small, objectives=objs)
    items = _all_ok(plan)
    random.Random(7).shuffle(items)
    manifest, rows = _collect(plan, items, small, objs, tmp_path)
    trust = {o.id for o in objs if o.is_trust_escalation}
    for row in rows:
        turn = "prov" if row["objective_id"] in trust else "1"
        cid = f"{row['technique_id']}_{row['objective_id']}_t{row['trial']}_{turn}"
        assert row["response"] == f"reply for {cid}"
        if turn == "prov":
            assert row["anon_response"] == f"reply for {cid[:-4]}anon"
            assert row["placebo_response"] == f"reply for {cid[:-4]}plac"
    assert manifest["sampling"]["mode"] == "batch"
    assert manifest["sampling"]["unsupported_controls"] == ["seed", "temperature"]
    assert manifest["batch"]["succeeded"] == manifest["batch"]["planned"] == len(plan["requests"])
    assert manifest["batch"]["fallbacks_sent"] is False
    assert manifest["determinism"] is None


def test_collect_lists_errored_expired_and_missing(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    ids = [r["custom_id"] for r in plan["requests"] if r["custom_id"].endswith("_1")]
    errored, expired, missing = ids[0], ids[1], ids[2]
    items = [_errored(errored), _expired(expired), _ok("not-in-the-plan_t0_1")]
    items += [_ok(r["custom_id"]) for r in plan["requests"] if r["custom_id"] not in (errored, expired, missing)]
    manifest, rows = _collect(plan, items, small, objs, tmp_path)
    bt = manifest["batch"]
    assert (bt["errored"], bt["expired"], bt["missing"]) == ([errored], [expired], [missing])
    assert bt["unplanned"] == ["not-in-the-plan_t0_1"] and bt["canceled"] == []
    assert bt["succeeded"] == len(plan["requests"]) - 3
    by_id = {f"{r['technique_id']}_{r['objective_id']}_t{r['trial']}_1": r for r in rows}
    assert all(by_id[i]["status"] == "error" and verdict_of(by_id[i]) == "error"
               for i in (errored, expired, missing))
    md = build_report(rows, manifest=manifest)
    for i in (errored, expired, missing):
        assert f"`{i}`" in md
    assert "Collected from Message Batch" in md and "judge validation" in md


def test_collect_refuses_a_plan_the_code_no_longer_builds(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    plan["requests"][3]["params"]["max_tokens"] = 7
    out = tmp_path / "artifacts"
    with pytest.raises(b.PlanDrift, match=plan["requests"][3]["custom_id"]):
        b.collect_plan(plan, _results(_all_ok(plan)), techniques=small, objectives=objs,
                       artifacts_dir=out)
    assert not out.exists() or not list(out.iterdir())  # refused before any run, not after


def test_collect_refuses_a_changed_library(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    with pytest.raises(b.PlanDrift, match="technique library changed"):
        b.collect_plan(plan, {}, techniques=small[:1], objectives=objs, artifacts_dir=tmp_path)


def test_a_temperature_400_in_the_batch_fails_the_collection(small, objs, tmp_path):
    """A batch item refusing `temperature` fails the run, as on a synchronous call (ISC-145)."""
    plan = _plan(WITH_TEMP, techniques=small, objectives=objs)
    items = _all_ok(plan)
    items[0] = _errored(plan["requests"][0]["custom_id"], "invalid_request_error", DEPRECATED)
    with pytest.raises(SamplingRefused):
        b.collect_plan(plan, _results(items), techniques=small, objectives=objs, artifacts_dir=tmp_path)


def test_a_refused_batch_item_carries_the_vendor_label(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    cid = plan["requests"][0]["custom_id"]
    items = [_ok(r["custom_id"]) for r in plan["requests"][1:]]
    items.append(_ok(cid, text="", stop_reason="refusal",
                     stop_details={"type": "refusal", "category": "cyber", "explanation": "x"}))
    _, rows = _collect(plan, items, small, objs, tmp_path)
    row = next(r for r in rows if r["trial"] == 0)
    assert (row["vendor_stop_reason"], row["vendor_stop_category"]) == ("refusal", "cyber")
    assert row["model_stratum"] == "requested" and row["status"] == "evaluated"


def test_batch_artifact_reports_like_a_sync_artifact(small, objs, tmp_path):
    """The same replies through the batch path and a synchronous run give the same rows (bar
    timing) and the same report body; regrade reads the batch artifact, and compare refuses it
    exactly as it refuses a synchronous chatbot artifact (compare reads agent artifacts)."""
    plan = _plan(techniques=small, objectives=objs)
    replies = {_key(r["params"]): f"reply {i}" for i, r in enumerate(plan["requests"])}
    batch_dir, sync_dir = tmp_path / "batch", tmp_path / "sync"
    b_manifest, b_rows = _collect(plan, [_ok(r["custom_id"], replies[_key(r["params"])])
                                         for r in plan["requests"]], small, objs, batch_dir)
    answering = _Answering(lambda kw: replies[_key(kw)])
    sync_path = run(AnthropicTarget(MODEL, client=_client(messages=answering), supported_options=NO_TEMP),
                    authorized=True, techniques=small, objectives=objs, trials=2, temperature=0.8,
                    base_seed=1337, artifacts_dir=sync_dir, determinism_check=False)
    s_manifest, s_rows = read_artifact(sync_path)
    volatile = {"timestamp", "latency_s"}
    assert ([{k: v for k, v in r.items() if k not in volatile} for r in b_rows]
            == [{k: v for k, v in r.items() if k not in volatile} for r in s_rows])

    def body(md, manifest):
        disclosure = batch_disclosure(manifest)
        md = md.replace(disclosure + "\n\n", "") if disclosure else md
        return [ln for ln in md.splitlines() if not re.search(r"\d{4}-\d\d-\d\dT", ln)]

    assert batch_disclosure(b_manifest) and not batch_disclosure(s_manifest)
    assert (body(build_report(b_rows, manifest=b_manifest), b_manifest)
            == body(build_report(s_rows, manifest=s_manifest), s_manifest))
    batch_path = next(batch_dir.glob("*.jsonl"))
    for path in (batch_path, sync_path):
        with pytest.raises(ValueError, match=r"holds \['chatbot'\] rows"):
            build_comparison([path, path])

    class Judge:
        def judge(self, objective, response, kind="forbidden"):
            from iago.judge import REFUSED, Verdict
            return Verdict(REFUSED, 0.9, "stub")

    assert regrade_file(batch_path, Judge())["regraded"] > 0


# --- the CLI collect loop ------------------------------------------------------------------------

def _batch(batch_id, status="ended"):
    return MessageBatch.model_validate({
        "id": batch_id, "type": "message_batch", "processing_status": status,
        "created_at": "2026-11-01T00:00:00Z", "expires_at": "2026-11-02T00:00:00Z",
        "ended_at": "2026-11-01T01:00:00Z" if status == "ended" else None,
        "archived_at": None, "cancel_initiated_at": None, "results_url": None,
        "request_counts": {"canceled": 0, "errored": 0, "expired": 0, "processing": 0, "succeeded": 0}})


def _collect_client(plans_by_id, status="ended", items=None):
    """A fake batches client: `retrieve` returns SDK `MessageBatch` models; `results` streams
    `items[batch_id]` when given, else a success for every planned request."""
    class Batches:
        def retrieve(self, batch_id):
            return _batch(batch_id, status)

        def results(self, batch_id):
            return iter((items or {}).get(batch_id) or _all_ok(plans_by_id[batch_id]))

    return _client(batches=Batches())


@pytest.fixture(scope="module")
def cli_plan():
    """A 3-technique, 1-trial plan the CLI collect tests share (each takes a copy)."""
    return _plan(trials=1, technique_limit=3)


def _record(cli_plan, tmp_path, *batch_ids):
    plans = []
    for batch_id in batch_ids or ("msgbatch_x",):
        plan = copy.deepcopy(cli_plan)
        plan["batch_id"] = batch_id
        plans.append(plan)
    return plans, b.write_record(plans, tmp_path / "plan.json")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("iago.runner.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("iago.report.REPORTS_DIR", tmp_path / "reports")
    return tmp_path / "artifacts"


def test_cli_collect_writes_artifact_and_report(cli_plan, tmp_path, monkeypatch, capsys, dirs):
    (plan,), record = _record(cli_plan, tmp_path, "msgbatch_x")
    monkeypatch.setattr(cli, "_batch_client", lambda **kw: _collect_client({"msgbatch_x": plan}))
    assert cli.main(["batch-collect", str(record), "--html"]) == 0
    out = capsys.readouterr().out
    n = len(plan["requests"])
    assert f"{n}/{n} succeeded" in out and "HTML:" in out
    (artifact,) = dirs.glob("*.jsonl")
    manifest = json.loads(artifact.read_text().splitlines()[0])
    assert manifest["batch"]["batch_id"] == "msgbatch_x"
    assert manifest["batch"]["ended_at"].startswith("2026-11-01 01:00:00")


def test_cli_collect_reports_a_batch_still_processing(cli_plan, tmp_path, monkeypatch, capsys):
    (plan,), record = _record(cli_plan, tmp_path, "msgbatch_y")
    monkeypatch.setattr(cli, "_batch_client",
                        lambda **kw: _collect_client({"msgbatch_y": plan}, status="in_progress"))
    assert cli.main(["batch-collect", str(record)]) == 1
    assert "still processing" in capsys.readouterr().out


def test_cli_collect_refuses_a_dry_run_record(cli_plan, tmp_path, capsys):
    record = b.write_record([cli_plan], tmp_path / "plan.json")
    assert cli.main(["batch-collect", str(record)]) == 2
    assert "no submitted batch" in capsys.readouterr().err


def test_cli_collect_batch_id_filter(cli_plan, tmp_path, monkeypatch, capsys, dirs):
    plans, record = _record(cli_plan, tmp_path, "msgbatch_a", "msgbatch_b")
    monkeypatch.setattr(cli, "_batch_client",
                        lambda **kw: _collect_client({p["batch_id"]: p for p in plans}))
    assert cli.main(["batch-collect", str(record), "--batch-id", "msgbatch_b"]) == 0
    (artifact,) = dirs.glob("*.jsonl")
    assert json.loads(artifact.read_text().splitlines()[0])["batch"]["batch_id"] == "msgbatch_b"
    assert cli.main(["batch-collect", str(record), "--batch-id", "nope"]) == 2


def test_cli_collect_failure_exits_2_and_still_collects_the_others(cli_plan, tmp_path, monkeypatch,
                                                                   capsys, dirs):
    """A drifted plan and a duplicated result id each exit 2 with an ERROR line, write no artifact
    of their own, and do not stop the next batch from being collected."""
    plans, record = _record(cli_plan, tmp_path, "msgbatch_drift", "msgbatch_ok")
    plans[0]["requests"][0]["params"]["max_tokens"] = 7
    b.write_record(plans, record)
    monkeypatch.setattr(cli, "_batch_client",
                        lambda **kw: _collect_client({p["batch_id"]: p for p in plans}))
    assert cli.main(["batch-collect", str(record)]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and plans[0]["requests"][0]["custom_id"] in err
    (artifact,) = dirs.glob("*.jsonl")
    assert json.loads(artifact.read_text().splitlines()[0])["batch"]["batch_id"] == "msgbatch_ok"

    dup = _all_ok(plans[1]) + [_ok(plans[1]["requests"][0]["custom_id"])]
    monkeypatch.setattr(cli, "_batch_client", lambda **kw: _collect_client(
        {p["batch_id"]: p for p in plans}, items={"msgbatch_ok": dup}))
    assert cli.main(["batch-collect", str(record), "--batch-id", "msgbatch_ok"]) == 2
    assert "twice" in capsys.readouterr().err


def test_wait_for_batch_polls_until_ended():
    statuses = iter(["in_progress", "in_progress", "ended"])
    client = _client(batches=SimpleNamespace(retrieve=lambda batch_id: _batch(batch_id, next(statuses))))
    slept = []
    batch = b.wait_for_batch(client, "x", wait=True, poll_seconds=3, sleep=slept.append)
    assert batch.processing_status == "ended" and slept == [3, 3]
    client = _client(batches=SimpleNamespace(retrieve=lambda batch_id: _batch(batch_id, "in_progress")))
    assert b.wait_for_batch(client, "x", wait=False, poll_seconds=3, sleep=slept.append) is None


# --- review findings (Council council-20261101T161334-zf8s9z, code review, test analysis) --------

def _canceled(cid):
    return MessageBatchIndividualResponse.model_validate(
        {"custom_id": cid, "result": {"type": "canceled"}})


def test_every_outcome_is_bucketed_and_named_in_the_report(small, objs, tmp_path):
    """canceled, an unknown result type (bucketed errored), and a thinking-only success (no_text)
    each land in their bucket, as RUN-ERROR rows, and in the report's not-answered sentence."""
    plan = _plan(techniques=small, objectives=objs)
    ids = [r["custom_id"] for r in plan["requests"] if r["custom_id"].endswith("_1")]
    canceled, unknown, thinking = ids[0], ids[1], ids[2]
    items = [_canceled(canceled),
             _ok(thinking, content=[{"type": "thinking", "thinking": "", "signature": "s"}],
                 stop_reason="max_tokens")]
    items += [_ok(r["custom_id"]) for r in plan["requests"] if r["custom_id"] not in (canceled, unknown, thinking)]
    results = _results(items)
    results[unknown] = SimpleNamespace(type="something_new")
    manifest, rows = read_artifact(b.collect_plan(plan, results, techniques=small, objectives=objs,
                                                  artifacts_dir=tmp_path))
    bt = manifest["batch"]
    assert (bt["canceled"], bt["errored"], bt["no_text"]) == ([canceled], [unknown], [thinking])
    assert bt["succeeded"] == len(plan["requests"]) - 3
    by_id = {f"{r['technique_id']}_{r['objective_id']}_t{r['trial']}_1": r for r in rows}
    assert all(by_id[i]["status"] == "error" for i in (canceled, unknown, thinking))
    disclosure = batch_disclosure(manifest)
    assert f"canceled `{canceled}`" in disclosure and f"no_text `{thinking}`" in disclosure
    assert f"errored `{unknown}`" in disclosure
    assert disclosure.startswith("_**Single-shot techniques only")


def test_an_unrelated_invalid_request_is_a_row_error_not_a_failed_run(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    items = _all_ok(plan)
    items[0] = _errored(plan["requests"][0]["custom_id"], "invalid_request_error", "model: not found")
    _, rows = _collect(plan, items, small, objs, tmp_path)
    assert sum(r["status"] == "error" for r in rows) == 1


def test_an_errored_trust_arm_leaves_every_other_row_on_its_own_reply(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    anon = next(r["custom_id"] for r in plan["requests"] if r["custom_id"].endswith("_anon"))
    items = [_errored(r["custom_id"]) if r["custom_id"] == anon else _ok(r["custom_id"])
             for r in plan["requests"]]
    _, rows = _collect(plan, items, small, objs, tmp_path)
    trust = {o.id for o in objs if o.is_trust_escalation}
    for row in rows:
        turn = "prov" if row["objective_id"] in trust else "1"
        assert row["response"] == f"reply for {row['technique_id']}_{row['objective_id']}_t{row['trial']}_{turn}"


def test_collect_refuses_a_plan_with_a_request_missing(small, objs, tmp_path):
    plan = _plan(techniques=small, objectives=objs)
    plan["requests"].pop()
    with pytest.raises(b.PlanDrift, match="holds"):
        b.collect_plan(plan, {}, techniques=small, objectives=objs, artifacts_dir=tmp_path)


def test_a_call_out_of_step_with_the_plan_quarantines_the_artifact(small, objs, tmp_path, monkeypatch):
    """Defence in depth behind the re-render check: if the collection's calls ever stop matching
    the plan in order, the run fails and its rows are renamed out of every `*.jsonl` reader's way."""
    plan = _plan(techniques=small, objectives=objs)
    truth = copy.deepcopy(plan)
    plan["requests"][2]["params"]["max_tokens"] = 7
    monkeypatch.setattr(b, "plan_requests", lambda *a, **k: plan)  # skip the re-render check
    with pytest.raises(b.PlanDrift, match=re.escape(truth["requests"][2]["custom_id"])):
        b.collect_plan(plan, _results(_all_ok(truth)), techniques=small, objectives=objs,
                       artifacts_dir=tmp_path)
    assert not list(tmp_path.glob("*.jsonl")) and list(tmp_path.glob("*.jsonl.misaligned"))


def test_custom_id_rejects_ids_the_api_would_refuse():
    with pytest.raises(ValueError):
        b.custom_id("a_b", "obj", 0, "1")
    with pytest.raises(ValueError):
        b.custom_id("t" * 40, "o" * 30, 0, "1")


def test_the_plan_renders_identically_across_processes():
    """collect re-plans in another process, days later: hash order must not move the plan."""
    import subprocess
    import sys

    code = ("import json; from iago import batch as b; p = b.plan_requests('claude-haiku-5-5', "
            "['num_predict'], probe='t', sampling={'trials': 1, 'temperature': 0.8, 'base_seed': 1, "
            "'category': None, 'technique_limit': 4, 'objective_limit': None, 'shots': None}); "
            "print(json.dumps(p['requests'], sort_keys=True))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**__import__("os").environ, "PYTHONHASHSEED": seed}).stdout
            for seed in ("1", "2")}
    assert len(outs) == 1 and json.loads(outs.pop())


def test_ceiling_counts_bytes_and_max_tokens_exactly():
    req = {"custom_id": "a", "params": {"system": "é" * 10, "max_tokens": 100,
                                        "messages": [{"role": "user", "content": "日本語"}]}}
    est = b.estimate([req], MODEL)["ceiling"]
    price_in, price_out = b.BATCH_PRICES[MODEL]
    assert est["input_tokens"] == 20 + 9 + b.REQUEST_OVERHEAD_TOKENS
    assert est["output_tokens"] == 100
    assert est["dollars"] == round(est["input_tokens"] * price_in / 1e6 + 100 * price_out / 1e6, 4)


def test_live_over_budget_creates_no_batch(tmp_path, monkeypatch, capsys):
    class Batches:
        def create(self, requests):
            pytest.fail("a batch was created over budget")

    monkeypatch.setattr(cli, "_batch_client", lambda **kw: _client(batches=Batches()))
    monkeypatch.setattr(cli, "build_target",
                        lambda kind, model: SimpleNamespace(supported_options=frozenset(NO_TEMP)))
    out = tmp_path / "plan.json"
    assert cli.main(["batch-submit", "--model", MODEL, "--max-dollars", "0.0001", "--authorized",
                     "--out", str(out), *SMALL_CLI]) == 2
    assert "no batch was created" in capsys.readouterr().err and not out.exists()


def test_the_budget_is_the_sum_over_models_with_an_inclusive_cap(tmp_path, capsys):
    plans = [_plan(model=m, trials=1, technique_limit=3) for m in (MODEL, "claude-opus-5-5")]
    each, total = [b.total_ceiling([p]) for p in plans], b.total_ceiling(plans)
    args = ["batch-submit", "--model", MODEL, "--model", "claude-opus-5-5", "--dry-run", *SMALL_CLI]
    under_sum = max(each) + (total - max(each)) / 2
    assert cli.main([*args, "--max-dollars", str(under_sum), "--out", str(tmp_path / "a.json")]) == 2
    assert cli.main([*args, "--max-dollars", str(total), "--out", str(tmp_path / "b.json")]) == 0


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-5"])
def test_a_meaningless_budget_is_refused(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["batch-submit", "--model", MODEL, "--dry-run", "--max-dollars", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv, message", [
    (["--model", MODEL, "--model", MODEL], "repeated"),
    (["--model", "claude-unpriced-9"], "no Batch price"),
])
def test_submit_refuses_before_any_probe(argv, message, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: pytest.fail("probed before refusing"))
    assert cli.main(["batch-submit", *argv, "--max-dollars", "5", "--authorized"]) == 2
    assert message in capsys.readouterr().err


def test_submit_refuses_an_existing_record(tmp_path, monkeypatch, capsys):
    out = tmp_path / "plan.json"
    out.write_text("{}")
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: pytest.fail("probed before refusing"))
    assert cli.main(["batch-submit", "--model", MODEL, "--max-dollars", "5", "--authorized",
                     "--out", str(out)]) == 2
    assert "already exists" in capsys.readouterr().err and out.read_text() == "{}"


def test_a_failed_second_submit_keeps_the_first_batch_id(tmp_path, monkeypatch):
    created, options = [], {}

    class Batches:
        def create(self, requests):
            if created:
                raise RuntimeError("APITimeoutError")
            created.append(requests)
            return _batch("msgbatch_first", "in_progress")

    def client(**kw):
        options.update(kw)
        return _client(batches=Batches())

    monkeypatch.setattr(cli, "_batch_client", client)
    monkeypatch.setattr(cli, "build_target",
                        lambda kind, model: SimpleNamespace(supported_options=frozenset(NO_TEMP)))
    out = tmp_path / "plan.json"
    with pytest.raises(RuntimeError):
        cli.main(["batch-submit", "--model", MODEL, "--model", "claude-opus-5-5", "--max-dollars", "5",
                  "--authorized", "--out", str(out), *SMALL_CLI])
    first, second = b.read_record(out)
    assert first["batch_id"] == "msgbatch_first" and "batch_id" not in second
    assert options == {"max_retries": 0}
