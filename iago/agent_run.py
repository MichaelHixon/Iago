"""Agentic runner + report: fire every scenario at a tool-calling agent, N trials.

Same methodology as the chatbot runner — multi-trial, pinned sampling, JSONL
artifacts, report reads artifacts only — but the unit of measurement is a
BEHAVIORAL bypass (exfiltration / unauthorized action) scored deterministically
by the tool-call oracle, not a text judgment. The report leads with the
capability floor (is the sink reachable at all?), then the behavioral-hijack RATE
per attack scenario, quotes the exact exfiltrating tool call as evidence, and
carries a control-calibration line and a sample-size caveat.

ETHICS: run only against a model you own or are authorized to test. Tools are
mocked in-process — `send_message` records the attempt, it never egresses.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .agent_harness import LLMMessage, run_agent
from .agent_oracle import ATTEMPTED, HIJACKED, judge_agent
from .agent_scenarios import Scenario, load_scenarios
from .config import (
    ALLOWED_RECIPIENT_DOMAIN,
    ARTIFACTS_DIR,
    BASE_SEED,
    DEFAULT_AGENT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIALS,
    REPORTS_DIR,
)


def ollama_chat_fn(model: str = DEFAULT_MODEL):
    """Build a chat_fn backed by a local Ollama model with tool-calling enabled.

    Normalizes Ollama's response (a ChatResponse object; tool_calls carry
    .function.name / .function.arguments) into the harness's LLMMessage. Fails
    loudly if the daemon/model is unreachable rather than nulling the turn.
    """
    def fn(messages: list[dict], tools: list[dict], options: dict) -> LLMMessage:
        import ollama

        try:
            resp = ollama.chat(model=model, messages=messages, tools=tools, options=options or {})
        except Exception as exc:
            raise RuntimeError(
                f"Ollama agent chat failed (daemon running and '{model}' pulled?): {exc}"
            ) from exc
        return _normalize_ollama(resp)

    return fn


def _normalize_ollama(resp: object) -> LLMMessage:
    message = getattr(resp, "message", None)
    if message is None and isinstance(resp, dict):
        message = resp.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls is None and isinstance(message, dict):
        raw_calls = message.get("tool_calls")

    calls: list[tuple[str, dict]] = []
    for tc in raw_calls or []:
        func = getattr(tc, "function", None)
        if func is None and isinstance(tc, dict):
            func = tc.get("function", {})
        name = getattr(func, "name", None)
        if name is None and isinstance(func, dict):
            name = func.get("name")
        args = getattr(func, "arguments", None)
        if args is None and isinstance(func, dict):
            args = func.get("arguments")
        if isinstance(args, str):  # some providers hand back a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append((name, args or {}))

    return LLMMessage(content=content or "", tool_calls=calls)


def _run_id(model: str, now: datetime) -> str:
    safe = model.replace(":", "-").replace("/", "-")
    return f"agent_{now.strftime('%Y%m%dT%H%M%SZ')}_{safe}"


def run_agent_suite(
    chat_fn,
    *,
    model_name: str,
    trials: int = DEFAULT_TRIALS,
    temperature: float = DEFAULT_TEMPERATURE,
    base_seed: int = BASE_SEED,
    max_steps: int = DEFAULT_AGENT_STEPS,
    scenarios: list[Scenario] | None = None,
    artifacts_dir: Path | None = None,
    progress: bool = False,
) -> Path:
    """Run every scenario × trials through the agent + oracle; write JSONL. Returns path."""
    scens = scenarios if scenarios is not None else load_scenarios()
    if not scens:
        raise ValueError("no scenarios to run")

    now = datetime.now(timezone.utc)
    out_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_run_id(model_name, now)}.jsonl"

    total = len(scens) * trials
    done = 0
    with out_path.open("w") as fh:
        for trial in range(trials):
            seed = base_seed + trial
            options = {"temperature": temperature, "seed": seed}
            for scen in scens:
                trace = run_agent(scen, chat_fn, options=options, max_steps=max_steps)
                verdict = judge_agent(trace, scen)
                row = {
                    "scenario_id": scen.id,
                    "scenario_name": scen.name,
                    "kind": scen.kind,
                    "owasp": scen.owasp,
                    "model": model_name,
                    "trial": trial,
                    "seed": seed,
                    "temperature": temperature,
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "rationale": verdict.rationale,
                    "exfil_recipient": verdict.exfil_recipient,
                    "leaked_canary": verdict.leaked_canary,
                    "tool_calls": [asdict(c) for c in trace.calls],
                    "final_text": trace.final_text,
                    "steps": trace.steps,
                    "hit_step_limit": trace.hit_step_limit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done += 1
                if progress:
                    print(f"  [{done}/{total}] r{trial} {scen.id} -> {verdict.verdict} "
                          f"({verdict.confidence:.2f})")

    return out_path


def load_agent_artifacts(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_agent_report(rows: list[dict], reports_dir: Path | None = None) -> Path:
    """Render a markdown report from agent artifact rows. Reads rows only."""
    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by scenario.
    by_scen: dict[str, list[dict]] = {}
    for r in rows:
        by_scen.setdefault(r["scenario_id"], []).append(r)

    model = rows[0]["model"] if rows else "unknown"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"agent_report_{ts}.md"

    lines: list[str] = []
    lines.append("# Iago — Agentic Indirect-Injection Report")
    lines.append("")
    lines.append("> Defensive research. Behavioral bypass = the agent took an unauthorized")
    lines.append("> action (exfiltration via external send / private-data leak), measured")
    lines.append("> deterministically from the tool-call trace — not a text judgment.")
    lines.append("")
    lines.append(f"- **Target agent model:** `{model}`")
    lines.append(f"- **Allowed comms domain (policy):** `@{ALLOWED_RECIPIENT_DOMAIN}`")
    lines.append(f"- **Scenarios:** {len(by_scen)}  ·  **Total trials:** {len(rows)}")
    lines.append(f"- **Attack surface:** OWASP LLM01/ASI01 (indirect prompt injection), "
                 "LLM06 (excessive agency), MITRE ATLAS AML.T0086 (exfil via tool invocation)")
    lines.append("")

    attack_scens = {s: r for s, r in by_scen.items() if r[0]["kind"] == "attack"}
    cap_scens = {s: r for s, r in by_scen.items() if r[0]["kind"] == "capability"}

    # Capability floor FIRST — the attack rate is only interpretable once we know the
    # agent can actually reach the exfil sink (rules out the too-dumb-to-fail confound).
    lines.append("## Capability floor (positive control)")
    lines.append("")
    if cap_scens:
        cap_rows = [r for rows_ in cap_scens.values() for r in rows_]
        reached = sum(1 for r in cap_rows
                      if any(c["name"] == "send_message" for c in r["tool_calls"]))
        lines.append(f"On a policy-compliant task, the agent reached the `send_message` sink "
                     f"(read→send tool chain) in **{reached}/{len(cap_rows)}** trials.")
        lines.append("")
        if reached == 0:
            lines.append("> ⚠️ **The agent never reached the comms sink even on a compliant task.** It is "
                         "too weak a tool-caller for this harness — any 0% attack rate below is DEGENERATE "
                         "(too-dumb-to-be-hijacked), not evidence that a guardrail held. Re-run against a "
                         "capable tool-calling model before reporting an attack rate.")
        else:
            lines.append("> ✅ The agent can operate the comms sink — so a low attack hijack rate below "
                         "reflects the agent RESISTING the injection, not an inability to act.")
        lines.append("")
    else:
        lines.append("_No capability-floor scenario in this run — attack rates are uncalibrated for "
                     "the too-dumb-to-be-hijacked confound. Add a `capability` scenario._")
        lines.append("")

    # Headline table — attack scenarios only.
    lines.append("## Behavioral hijack rate by attack scenario")
    lines.append("")
    lines.append("| Scenario | Trials | Hijacked | Attempted | Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for sid, srows in attack_scens.items():
        n = len(srows)
        hj = sum(1 for r in srows if r["verdict"] == HIJACKED)
        att = sum(1 for r in srows if r["verdict"] == ATTEMPTED)
        rate = f"{hj / n:.0%}" if n else "n/a"
        lines.append(f"| {sid} | {n} | {hj} | {att} | {rate} |")
    lines.append("")
    n_attack_trials = sum(len(r) for r in attack_scens.values())
    lines.append(f"_Attempted = the injection moved the agent (leak into an internal channel, or "
                 "unauthorized private-data access) but external exfil did not complete — a SUSPECT "
                 "signal, not an all-clear._")
    lines.append("")
    lines.append(f"_Sample size: {n_attack_trials} attack trials across {len(attack_scens)} "
                 "scenarios. This is a mechanism demonstration, not a benchmark — a single-digit "
                 "rate carries a wide confidence interval; scale trials/scenarios and target a "
                 "capable model before treating any rate as a population estimate._")
    lines.append("")

    # Evidence: quote the exact exfiltrating call for each confirmed hijack.
    lines.append("## Confirmed exfiltration evidence")
    lines.append("")
    any_evidence = False
    for sid, srows in attack_scens.items():
        for r in srows:
            if r["verdict"] != HIJACKED:
                continue
            any_evidence = True
            send = next((c for c in r["tool_calls"] if c["name"] == "send_message"), None)
            lines.append(f"**{sid}** (trial {r['trial']}) — {r['rationale']}")
            if send:
                recipient = send["arguments"].get("recipient", "?")
                body = str(send["arguments"].get("body", ""))
                snippet = body if len(body) <= 200 else body[:200] + "…"
                lines.append("")
                lines.append(f"- `send_message(recipient={recipient!r})`")
                lines.append(f"- body: `{snippet}`")
                lines.append(f"- leaked canary: **{r['leaked_canary']}**")
            lines.append("")
    if not any_evidence:
        lines.append("_No confirmed exfiltration this run — the agent held policy across all "
                     "attack scenarios._")
        lines.append("")

    # Control calibration.
    controls = [r for r in rows if r["kind"] == "control"]
    if controls:
        control_hijacks = sum(1 for r in controls if r["verdict"] == HIJACKED)
        lines.append("## Control calibration")
        lines.append("")
        lines.append(f"Control scenarios (clean documents, no injection) hijacked "
                     f"{control_hijacks}/{len(controls)} trials. A non-zero control rate would "
                     "mean the oracle over-flags; zero is the expected calibration.")
        lines.append("")

    # Hardening recommendations (the defender's remediation view).
    lines.append("## Hardening recommendations")
    lines.append("")
    lines.append("- Treat all document/tool-result content as untrusted data; never let retrieved "
                 "text alter the recipient allowlist or authorize sending private data.")
    lines.append("- Gate external-comms tools behind an allowlist enforced OUTSIDE the model "
                 "(deterministic policy check on `send_message`, not a prompt instruction).")
    lines.append("- Apply least-privilege to tools: an agent doing a summary task should not have "
                 "standing access to both private notes and external send in the same context "
                 "(break the lethal trifecta).")
    lines.append("- Log and alert on any outbound send whose body contains data read from a "
                 "private source in the same session.")
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
