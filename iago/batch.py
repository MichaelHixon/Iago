"""Batch sweep: the runner's own plan sent through the Message Batches API, and the results read
back into an artifact in the runner's shape (ISC-148..150).

Both halves drive the real runner, so neither re-implements it:

  * `plan_requests` runs `runner.run` against an `AnthropicTarget` whose client only RECORDS each
    Messages call (no network), so every request in the plan is byte-for-byte what a synchronous
    run would have sent: the same `_fires` pairing, trials, per-objective system prompt, and
    `_create` parameters, with `temperature` omitted when the model refused it at the build probe.
  * `collect_plan` first re-plans with the plan's own settings and refuses a plan the current code
    or library no longer renders identically (`PlanDrift`). It then runs `runner.run` again against
    a client that answers the i-th call with the result filed under the i-th planned `custom_id`
    (results arrive in any order; they are looked up by id, never by position in the stream).

Multi-turn techniques are excluded: a priming turn's reply has to exist before the final request
can be written, so they cannot be one batch item each. Trust-escalation objectives fire three
planted single calls per trial (anonymous, placebo, provenance), so they are three items each.

`custom_id` is `<technique>_<objective>_t<trial>_<turn>`, where `<turn>` is `1` for a single-shot
call and `anon` / `plac` / `prov` for the three trust-escalation arms. The Batches API takes
`[A-Za-z0-9_-]{1,64}`; library ids use `-` and never `_`, so the id splits back unambiguously.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .artifacts import library_fingerprint, load_rows
from .attacks import Technique, load_library
from .objectives import Objective, load_objectives
from .runner import run
from .target import AnthropicTarget, _field, unsupported_options

PLAN_RECORD = "iago-batch-plan"
PLAN_FORMAT = 1

#: Message Batches prices, USD per million tokens (input, output): half the standard rate. Read
#: from https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-01 (the temperature
#: plan's § Cost estimate) and the claude-api skill's model table. A model missing here has no
#: estimate, so `batch-submit` refuses it rather than guessing.
BATCH_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.05, 0.25),
    "claude-sonnet-5-5": (1.00, 5.00),
    "claude-opus-5-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (0.50, 2.50),
}

#: The expected case (the plan's middle row): about 4 characters per input token and 250 output
#: tokens per reply. Printed for scale only; the budget check uses the ceiling.
EXPECTED_CHARS_PER_TOKEN = 4
EXPECTED_OUTPUT_TOKENS = 250
#: Message framing the character count does not see (role markers, the system slot).
REQUEST_OVERHEAD_TOKENS = 16

_CUSTOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SINGLE_TURN = "1"
TRUST_ARMS = ("anon", "plac", "prov")  # the order the runner fires them in
#: Ways a planned request can end without a judged reply, in the order the manifest and report list
#: them. `no_text` is a `succeeded` result with no text and no refusal (e.g. a reply that spent its
#: `max_tokens` on thinking): the target rejects it exactly as on a synchronous call, so its row is a
#: RUN-ERROR, and it is counted here rather than as succeeded.
OUTCOMES = ("errored", "expired", "canceled", "no_text", "missing")
NOT_ANSWERED = (*OUTCOMES, "unplanned")


class PlanDrift(ValueError):
    """The current code or library no longer renders the requests a plan was submitted with, so
    its results cannot be attributed to rows."""


def custom_id(technique_id: str, objective_id: str, trial: int, turn: str) -> str:
    cid = f"{technique_id}_{objective_id}_t{trial}_{turn}"
    if not _CUSTOM_ID.fullmatch(cid) or "_" in technique_id or "_" in objective_id:
        raise ValueError(f"cannot form a batch custom_id from {cid!r}: the Batches API takes "
                         "[A-Za-z0-9_-]{1,64} and library ids must not contain '_'")
    return cid


class _RecordedMessages:
    """`client.messages` for the planning pass: records every call, answers with a stub reply."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": [{"type": "text", "text": "planned"}], "stop_reason": "end_turn",
                "stop_details": None, "model": kwargs.get("model")}


def _single_shot(techniques: list[Technique] | None) -> tuple[list[Technique], list[str]]:
    lib = techniques if techniques is not None else load_library()
    return [t for t in lib if not t.is_multiturn], sorted(t.id for t in lib if t.is_multiturn)


def _run_kwargs(sampling: dict) -> dict:
    """The plan's sampling block as `runner.run` keyword arguments (the keys are its parameter
    names; `unsupported_controls` is a record of the result, not an input)."""
    return {k: v for k, v in sampling.items() if k != "unsupported_controls"}


def plan_requests(model: str, supported_options, *, probe: str, sampling: dict,
                  techniques: list[Technique] | None = None,
                  objectives: list[Objective] | None = None) -> dict:
    """The batch plan for one model: every single-shot call the runner fires, keyed by custom_id.
    `sampling` holds `runner.run`'s trials, temperature, base_seed, category, technique_limit,
    objective_limit and shots; `probe` records where `supported_options` came from."""
    lib, excluded = _single_shot(techniques)
    objs = objectives if objectives is not None else load_objectives()
    recorder = _RecordedMessages()
    target = AnthropicTarget(model, client=SimpleNamespace(messages=recorder),
                             supported_options=supported_options)
    with tempfile.TemporaryDirectory() as tmp:
        path = run(target, authorized=True, techniques=lib, objectives=objs,
                   artifacts_dir=Path(tmp), determinism_check=False, **_run_kwargs(sampling))
        rows = load_rows(path)

    trust = {o.id for o in objs if o.is_trust_escalation}
    calls, requests, seen = recorder.calls, [], set()
    for r in rows:
        turns = TRUST_ARMS if r["objective_id"] in trust else (SINGLE_TURN,)
        mine = calls[len(requests):len(requests) + len(turns)]
        asks = [c["messages"][-1]["content"] for c in mine]
        # Each key must sit on the call the runner made for it: a single-shot call's user turn is
        # the row's prompt, and a trust row's three arms all end on the same ask.
        if len(asks) != len(turns) or len(set(asks)) != 1 or (len(turns) == 1 and asks[0] != r["prompt"]):
            raise RuntimeError(f"planned calls are out of step with the runner at "
                               f"{r['technique_id']} x {r['objective_id']} trial {r['trial']}")
        for turn, params in zip(turns, mine):
            cid = custom_id(r["technique_id"], r["objective_id"], r["trial"], turn)
            if cid in seen:
                raise RuntimeError(f"two planned calls share the custom_id {cid}")
            if "fallbacks" in params:  # the Batches API errors such an item; Iago never opts in
                raise RuntimeError(f"{cid}: a planned request carries `fallbacks`")
            seen.add(cid)
            requests.append({"custom_id": cid, "params": params})
    if len(requests) != len(calls):
        raise RuntimeError(f"the planning run made {len(calls)} calls for {len(requests)} planned "
                           "items; the runner's call pattern changed and batch.py must follow it")

    dropped = unsupported_options(target, {"temperature": sampling["temperature"],
                                           "seed": sampling["base_seed"]})
    return {"model": model, "supported_options": sorted(target.supported_options), "probe": probe,
            "sampling": {**sampling, "unsupported_controls": dropped},
            "technique_library_sha256": library_fingerprint(lib), "excluded_multiturn": excluded,
            "requests": requests, "estimate": estimate(requests, model)}


def _text_sizes(params: dict) -> tuple[int, int]:
    """(characters, UTF-8 bytes) of everything the request sends as text."""
    texts = [params.get("system") or ""] + [str(m["content"]) for m in params["messages"]]
    return sum(len(t) for t in texts), sum(len(t.encode()) for t in texts)


def estimate(requests: list[dict], model: str) -> dict:
    """The plan's cost at Batch prices. `ceiling` is a bound, not a guess: input counted as one
    token per UTF-8 byte (no tokenizer emits more tokens than bytes) and output at each request's
    `max_tokens`, which caps thinking and text together. `expected` is the plan's middle case."""
    if model not in BATCH_PRICES:
        raise ValueError(f"no Batch price recorded for {model!r}; add it to batch.BATCH_PRICES "
                         "from the pricing page before submitting")
    price_in, price_out = BATCH_PRICES[model]

    def case(tokens_in: int, tokens_out: int) -> dict:
        return {"input_tokens": tokens_in, "output_tokens": tokens_out,
                "dollars": round(tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6, 4)}

    sizes = [_text_sizes(r["params"]) for r in requests]
    exp_in = sum(math.ceil(c / EXPECTED_CHARS_PER_TOKEN) + REQUEST_OVERHEAD_TOKENS for c, _ in sizes)
    ceil_in = sum(n + REQUEST_OVERHEAD_TOKENS for _, n in sizes)
    ceil_out = sum(int(r["params"]["max_tokens"]) for r in requests)
    return {"requests": len(requests), "batch_price_per_mtok": {"input": price_in, "output": price_out},
            "expected": case(exp_in, len(requests) * EXPECTED_OUTPUT_TOKENS),
            "ceiling": case(ceil_in, ceil_out)}


def total_ceiling(plans: list[dict]) -> float:
    return round(sum(p["estimate"]["ceiling"]["dollars"] for p in plans), 4)


def estimate_lines(plans: list[dict]) -> list[str]:
    lines = []
    for p in plans:
        e, x, c = p["estimate"], p["estimate"]["expected"], p["estimate"]["ceiling"]
        lines.append(f"  {p['model']}: {e['requests']} requests; expected ~{x['input_tokens']:,} in / "
                     f"{x['output_tokens']:,} out = ${x['dollars']:.2f}; ceiling "
                     f"{c['input_tokens']:,} in / {c['output_tokens']:,} out = ${c['dollars']:.2f}")
    expected = sum(p["estimate"]["expected"]["dollars"] for p in plans)
    lines.append(f"  total at Batch prices: expected ${expected:.2f}; ceiling ${total_ceiling(plans):.2f}")
    return lines


def submit_plan(plan: dict, client) -> None:
    """Create the plan's batch and stamp the plan with its id and submission time. `client` must not
    retry (`max_retries=0`): a create that timed out after the server accepted it would otherwise
    be sent again, creating a second billed batch whose id nothing records."""
    batch = client.messages.batches.create(
        requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in plan["requests"]])
    plan["batch_id"] = _field(batch, "id")
    plan["submitted_at"] = datetime.now(timezone.utc).isoformat()


def write_record(plans: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"record": PLAN_RECORD, "format": PLAN_FORMAT, "plans": plans},
                               indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_record(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("record") != PLAN_RECORD or data.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path} is not an iago batch plan (format {PLAN_FORMAT})")
    return data["plans"]


# --- collect -------------------------------------------------------------------------------------

def wait_for_batch(client, batch_id: str, *, wait: bool, poll_seconds: float,
                   sleep: Callable[[float], None] = time.sleep):
    """The batch once its processing has ended, else None; with `wait`, poll until it ends."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if _field(batch, "processing_status") == "ended":
            return batch
        if not wait:
            return None
        sleep(poll_seconds)


def read_results(client, batch_id: str) -> dict[str, object]:
    """`custom_id` -> result, from the results stream (which arrives in any order)."""
    results: dict[str, object] = {}
    for item in client.messages.batches.results(batch_id):
        cid = _field(item, "custom_id")
        if cid in results:
            raise RuntimeError(f"batch {batch_id} returned custom_id {cid!r} twice")
        results[cid] = _field(item, "result")
    return results


def account(plan: dict, results: dict[str, object]) -> dict[str, list[str]]:
    """Every planned id, accounted for: succeeded, errored, expired, canceled, or missing from the
    results; plus anything returned that was never planned. An unknown result type is errored."""
    out: dict[str, list[str]] = {k: [] for k in ("succeeded", *OUTCOMES)}
    planned = [r["custom_id"] for r in plan["requests"]]
    for cid in planned:
        kind = "missing" if cid not in results else _field(results[cid], "type")
        if kind == "succeeded" and _no_text(_field(results[cid], "message")):
            kind = "no_text"
        out[kind if kind in out else "errored"].append(cid)
    out["unplanned"] = sorted(set(results) - set(planned))
    return out


def _no_text(message: object) -> bool:
    """The case `AnthropicTarget._reply` rejects: no text in the content and no refusal."""
    blocks = _field(message, "content") or []
    text = "".join(_field(b, "text") or "" for b in blocks).strip()
    return not text and _field(message, "stop_reason") != "refusal"


class BatchItemError(RuntimeError):
    """An errored batch item, carrying the API's status shape so `is_temperature_refusal` reads a
    temperature 400 exactly as on a synchronous call (and the run fails rather than resampling)."""

    def __init__(self, cid: str, result: object) -> None:
        err = _field(_field(result, "error"), "error")
        etype, message = _field(err, "type"), _field(err, "message")
        super().__init__(f"batch item {cid} errored: {etype}: {message}")
        self.body = {"error": {"type": etype, "message": message or ""}}
        self.status_code = 400 if etype == "invalid_request_error" else None


class _ResultMessages:
    """`client.messages` for the collection pass: the i-th call gets the result filed under the
    i-th planned custom_id. `collect_plan` has already shown the plan re-renders identically; each
    call is still checked against its planned request, and a call that does not match (or one past
    the end of the plan) is answered with no reply and recorded in `misaligned`, which fails the
    collection after the run. Raising here would only become a RUN-ERROR row inside `_create`."""

    def __init__(self, plan: dict, results: dict[str, object]) -> None:
        self.requests = plan["requests"]
        self.results = results
        self.served = 0
        self.misaligned: list[str] = []

    def create(self, **kwargs):
        if self.served >= len(self.requests):
            self.misaligned.append(f"call {self.served + 1} is past the end of the plan")
            raise RuntimeError("call past the end of the plan")
        entry = self.requests[self.served]
        self.served += 1
        cid = entry["custom_id"]
        if kwargs != entry["params"]:
            self.misaligned.append(cid)
            raise RuntimeError(f"call does not match the request planned as {cid}")
        if cid not in self.results:
            raise RuntimeError(f"batch item {cid} missing from the results")
        res = self.results[cid]
        kind = _field(res, "type")
        if kind == "succeeded":
            return _field(res, "message")
        if kind == "errored":
            raise BatchItemError(cid, res)
        raise RuntimeError(f"batch item {cid} {kind}")


def collect_plan(plan: dict, results: dict[str, object], *, batch: object | None = None,
                 techniques: list[Technique] | None = None,
                 objectives: list[Objective] | None = None,
                 artifacts_dir: Path | None = None) -> Path:
    """Write the runner artifact for one collected batch. The rows come from the runner itself; a
    result that did not succeed becomes the same RUN-ERROR row a failed synchronous call does, and
    its id is listed in the manifest's `batch` block."""
    lib, excluded = _single_shot(techniques)
    objs = objectives if objectives is not None else load_objectives()
    if library_fingerprint(lib) != plan["technique_library_sha256"]:
        raise PlanDrift("the technique library changed since the plan was made")
    replanned = plan_requests(plan["model"], plan["supported_options"], probe=plan["probe"],
                              sampling=_run_kwargs(plan["sampling"]), techniques=lib, objectives=objs)
    for old, new in zip(plan["requests"], replanned["requests"]):
        if old != new:
            raise PlanDrift(f"{old['custom_id']}: the request the code builds today differs from "
                            "the one submitted; collect with the code the plan was made with")
    if len(plan["requests"]) != len(replanned["requests"]):
        raise PlanDrift(f"the plan holds {len(plan['requests'])} requests; the code renders "
                        f"{len(replanned['requests'])} today")
    acct = account(plan, results)
    ended = _field(batch, "ended_at")
    messages = _ResultMessages(plan, results)
    target = AnthropicTarget(plan["model"], client=SimpleNamespace(messages=messages),
                             supported_options=plan["supported_options"])
    target.manifest_extra = {"batch": {
        "batch_id": plan.get("batch_id"), "submitted_at": plan.get("submitted_at"),
        "ended_at": str(ended) if ended is not None else None,
        "planned": len(plan["requests"]), "succeeded": len(acct["succeeded"]),
        **{k: acct[k] for k in NOT_ANSWERED},
        "excluded_multiturn": excluded, "fallbacks_sent": False, "probe": plan["probe"],
    }}
    path = run(target, authorized=True, techniques=lib, objectives=objs, artifacts_dir=artifacts_dir,
               determinism_check=False, sampling_mode="batch", **_run_kwargs(plan["sampling"]))
    if messages.misaligned or messages.served != len(plan["requests"]):
        # Renamed, not deleted: kept for diagnosis, but no `*.jsonl` reader will pick it up.
        path = path.rename(path.with_name(path.name + ".misaligned"))
        raise PlanDrift(f"the collection run did not make the planned calls in order (misaligned: "
                        f"{messages.misaligned[:3]}; {messages.served} of {len(plan['requests'])} "
                        f"served); the rows are not a valid measurement and were moved to {path}")
    return path
