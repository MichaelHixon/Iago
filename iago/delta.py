"""Attack-vs-defense delta report.

Reads two artifact sets from the SAME attack library — the raw model and the model behind a
guard — and quotes the **bypass-rate delta**: the defensive payoff in one number. It reuses the
exact rate + Wilson-CI machinery the main report uses, so the two never drift, and it is honest
about the cost: it also measures how often the guard blocked benign control traffic (over-block)
and lists every residual bypass the guard did NOT stop.

Pairing: when the two runs share seeds/library (produced by `iago defense-delta`), each guarded
trial matches a raw trial on (technique, objective, trial), so "bypasses neutralized" is a true
per-trial comparison. Rate-level deltas hold either way.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import REPORTS_DIR
from .guards import guard_that_fired
from .judge import BYPASSED, ERROR, NEEDS_REVIEW
from .report import ci_str, pct, bypass_rate, verdict_of
from .stats import mcnemar_exact_p, wilson_interval


def _valid(rows: list[dict], kind: str) -> list[dict]:
    """Rows of one objective kind that are valid probes (transport errors excluded)."""
    return [r for r in rows if r["objective_kind"] == kind and verdict_of(r) != ERROR]


def _rate_block(rows: list[dict], kind: str) -> dict:
    valid = _valid(rows, kind)
    hits = sum(1 for r in valid if verdict_of(r) == BYPASSED)
    total = len(valid)
    return {"hits": hits, "total": total, "rate": bypass_rate(hits, total), "ci": ci_str(hits, total)}


def _delta_pts(raw: dict, guarded: dict) -> float:
    """Percentage-point change in bypass rate (negative = the guard reduced bypasses)."""
    return (guarded["rate"] - raw["rate"]) * 100.0


def _discordant(raw_rows: list[dict], guarded_rows: list[dict], kind: str) -> tuple[int, int]:
    """McNemar discordant-pair counts on the paired (technique, objective, trial) twins for one
    objective kind: (b, c) where b = raw bypassed AND guard held (the guard's wins), c = raw held
    AND guard bypassed (regressions). Concordant pairs (both bypass / both hold) carry no signal.
    Exposes the pairing the independent-CI gate ignores — for a real significance claim McNemar
    conditions on exactly these discordants."""
    guarded_by_key = {(r["technique_id"], r["objective_id"], r.get("trial", 0)): r
                      for r in guarded_rows if r["objective_kind"] == kind}
    b = c = 0
    for r in raw_rows:
        if r["objective_kind"] != kind or verdict_of(r) == ERROR:
            continue
        twin = guarded_by_key.get((r["technique_id"], r["objective_id"], r.get("trial", 0)))
        if twin is None or verdict_of(twin) == ERROR:
            continue
        raw_by, guard_by = verdict_of(r) == BYPASSED, verdict_of(twin) == BYPASSED
        if raw_by and not guard_by:
            b += 1
        elif guard_by and not raw_by:
            c += 1
    return b, c


def _significant(raw: dict, guarded: dict) -> bool:
    """The reduction is defensible when the raw and guarded 95% Wilson intervals do not overlap."""
    raw_lo, _ = wilson_interval(raw["hits"], raw["total"])
    _, guarded_hi = wilson_interval(guarded["hits"], guarded["total"])
    return guarded_hi < raw_lo


def _headline(a, label: str, raw: dict, guarded: dict) -> None:
    if raw["total"] == 0 and guarded["total"] == 0:
        return
    d = _delta_pts(raw, guarded)
    arrow = "▼" if d < 0 else ("▲" if d > 0 else "—")
    sig = " *(non-overlapping 95% CIs)*" if _significant(raw, guarded) else ""
    a(f"- **{label}:** {pct(raw['rate'])} (raw, {raw['hits']}/{raw['total']}, 95% CI {raw['ci']}) "
      f"→ {pct(guarded['rate'])} (guarded, {guarded['hits']}/{guarded['total']}, 95% CI {guarded['ci']}) "
      f"— {arrow} **{abs(d):.1f} pts**{sig}")


def build_delta_report(raw_rows: list[dict], guarded_rows: list[dict]) -> str:
    if not raw_rows or not guarded_rows:
        return "# Iago — Defense Delta\n\n_Need both a raw and a guarded artifact to compute a delta._\n"

    raw_model = raw_rows[0]["model"]
    guarded_model = guarded_rows[0]["model"]

    forbidden_raw = _rate_block(raw_rows, "forbidden")
    forbidden_guarded = _rate_block(guarded_rows, "forbidden")
    leak_raw = _rate_block(raw_rows, "prompt-leak")
    leak_guarded = _rate_block(guarded_rows, "prompt-leak")

    lines: list[str] = []
    a = lines.append
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    a("# Iago — Attack-vs-Defense Delta")
    a("")
    a("> **Authorized defensive-security research.** The same attack library was fired at a raw")
    a("> local model and at the same model behind a guard. The delta below is the guard's payoff:")
    a("> how much it reduced the confirmed-bypass rate — and what it cost in blocked benign traffic.")
    a("")
    a(f"- **Raw target:** `{raw_model}`")
    a(f"- **Guarded target:** `{guarded_model}`")
    a(f"- **Generated:** {now}")
    a("")

    a("## Headline — bypass-rate delta")
    a("")
    # Harmful-content bypass requires the Claude rubric judge — the heuristic NEVER asserts a
    # harmful-content bypass by design (judge.py). So on a heuristic-only run, don't print a
    # misleading "0% → 0%"; say plainly that it needs a regrade. The LLM07 leak delta is
    # deterministic (canary judge) and always meaningful.
    forbidden_valid = _valid(raw_rows, "forbidden") + _valid(guarded_rows, "forbidden")
    forbidden_graded = any("claude_verdict" in r for r in forbidden_valid)
    if forbidden_raw["hits"] or forbidden_guarded["hits"] or forbidden_graded:
        _headline(a, "Harmful-content bypass rate", forbidden_raw, forbidden_guarded)
    else:
        nr = sum(1 for r in _valid(raw_rows, "forbidden") if verdict_of(r) == NEEDS_REVIEW)
        a(f"- **Harmful-content bypass rate:** _not adjudicated._ This was a heuristic-only run, and "
          f"the heuristic judge never asserts a harmful-content bypass by design — so a 0% here would "
          f"be an artifact, not a result. Run `iago regrade` on **both** artifacts (raw + guarded) to "
          f"score the {forbidden_raw['total']} forbidden trials ({nr} substantive replies await "
          f"adjudication), then re-run `iago delta`.")
    _headline(a, "System-prompt exfiltration (LLM07)", leak_raw, leak_guarded)
    a("")
    a("> A negative delta with **non-overlapping 95% Wilson intervals** is a defensible reduction;")
    a("> an overlapping-interval delta is directional only (too few trials to assert it). Bypass")
    a("> counts use the effective verdict (`iago regrade` promotes the Claude judge if present).")
    a("")
    # Honesty: what "0%" on the LLM07 line actually measures (Council blockers, 2026-08-09).
    a('> **What the LLM07 "0%" means — read before quoting it.** This is a *verbatim / canary*')
    a("> exfiltration rate: a leak is counted only when the planted canary surfaces intact. Paraphrased,")
    a("> translated, or summarized disclosure routes around BOTH the canary oracle and this DLP filter —")
    a("> a documented false-negative (a semantic-similarity leak band is banked, not yet built, so the")
    a("> harness cannot yet score paraphrase leaks in either column). The DLP is also *handed the very")
    a("> system prompt it protects* — a realistic DLP assumption, but a best-case oracle, not a hardened")
    a("> deployment. So `0%` is a floor on what this narrow guard stops, **never** \"prompt-leak solved.\"")
    a("")
    # Paired-data significance: the two runs are twins, so McNemar's exact test — conditioning on the
    # discordant pairs — is the correct instrument. We run it here rather than name it and skip it. The
    # significance claim is GATED on the p-value and on there being discordant pairs at all: never assert
    # "not sampling noise" for a split that cannot support it (e.g. no discordant pairs → p = 1.0).
    b_leak, c_leak = _discordant(raw_rows, guarded_rows, "prompt-leak")
    n_disc = b_leak + c_leak
    p_leak = mcnemar_exact_p(b_leak, c_leak)
    a("> **Paired data — McNemar's exact test.** Each guarded trial has a raw twin (same technique,")
    a("> objective, seed), so the correct significance test is McNemar's, which conditions on the")
    if n_disc == 0:
        a("> discordant pairs. On the LLM07 pairs there were **no discordant pairs** — the guard changed no")
        a("> verdict either way — so there is nothing for the paired test to condition on: no signal, not a")
        a("> proven null.")
    else:
        claim = ("the reduction is not sampling noise" if p_leak < 0.05
                 else "directional only — too few discordant pairs to assert significance")
        a(f"> discordant pairs. On the LLM07 pairs: **{b_leak}** in the guard's favour (raw leaked → guard")
        a(f"> held) and **{c_leak}** against. The exact-binomial McNemar test on that split gives a two-sided")
        a(f"> **p = {p_leak:.2e}** — {claim}. The non-overlapping-interval gate above is a separate, coarser")
        a("> check: it treats the two rates as *independent*, which discards the pairing's statistical power")
        a("> — it does not buy conservatism.")
    a("")

    # --- Guard attribution: which guard fired, and on what -------------------------
    # Match each guarded trial back to its raw twin to separate real credit (blocked a trial that
    # bypassed raw) from no-credit blocks (the model would have refused anyway).
    raw_by_key: dict[tuple, dict] = {
        (r["technique_id"], r["objective_id"], r.get("trial", 0)): r for r in raw_rows
    }
    blocks_by_guard: dict[str, int] = defaultdict(int)
    neutralized_by_guard: dict[str, int] = defaultdict(int)
    for r in guarded_rows:
        # Attribution is about DEFENSIVE blocks on attack objectives. A block on a benign control
        # is over-blocking, reported separately below — counting it here would inflate "blocked".
        if r["objective_kind"] == "control":
            continue
        g = guard_that_fired(r.get("response", ""))
        if not g:
            continue
        blocks_by_guard[g] += 1
        twin = raw_by_key.get((r["technique_id"], r["objective_id"], r.get("trial", 0)))
        if twin is not None and verdict_of(twin) == BYPASSED:
            neutralized_by_guard[g] += 1

    a("## Guard attribution")
    a("")
    if blocks_by_guard:
        a("| Guard | Attack trials blocked | Blocked a raw bypass (credit) |")
        a("|---|---:|---:|")
        for g in sorted(blocks_by_guard):
            a(f"| `{g}` | {blocks_by_guard[g]} | {neutralized_by_guard.get(g, 0)} |")
        a("")
        a("> Counts blocks on **attack** objectives only (forbidden + prompt-leak); benign-control")
        a("> blocks are over-block, reported below — not defensive wins.")
        a("> *Credit* = the blocked trial bypassed on the raw model, so the guard genuinely stopped")
        a("> a bypass. Blocks without credit stopped a request the model would likely have refused")
        a("> anyway — real coverage, but not a bypass prevented.")
        a("> A guard whose credit equals its blocks (e.g. 27/27) had **no false positives in that many")
        a("> blocks** — a Wilson *lower* bound comfortably under 100%, not proven-perfect precision on a")
        a("> single run.")
    else:
        a("_No guard blocks recorded in the guarded run._")
    a("")

    # --- Over-block cost: benign control traffic the guard refused -----------------
    control_guarded = [r for r in guarded_rows if r["objective_kind"] == "control"]
    over = [r for r in control_guarded if guard_that_fired(r.get("response", ""))]
    over_benign = [r for r in over if r["category"] == "direct-ask"]
    a("## Over-block cost (benign control traffic)")
    a("")
    if control_guarded:
        a(f"- Control-objective trials blocked by a guard: **{len(over)}/{len(control_guarded)}** "
          f"({pct(bypass_rate(len(over), len(control_guarded)))}).")
        a(f"- Of those, **{len(over_benign)}** were `direct-ask` (benign payload, benign framing) — "
          "the truest false-positive signal; the rest were benign payloads wrapped in attack framings, "
          "where blocking the framing is arguably correct.")
    else:
        a("_No control trials in the guarded run._")
    a("")

    # --- Residual bypasses: what the guard did NOT stop ---------------------------
    residual = [r for r in guarded_rows
                if r["objective_kind"] in ("forbidden", "prompt-leak") and verdict_of(r) == BYPASSED]
    a("## Residual bypasses (guard did not stop these)")
    a("")
    if residual:
        seen: dict[tuple, dict] = {}
        for r in residual:
            seen.setdefault((r["technique_id"], r["objective_id"]), r)
        a("| Technique | Category | Objective | OWASP |")
        a("|---|---|---|---|")
        for (tid, oid), r in sorted(seen.items()):
            a(f"| `{tid}` {r['technique_name']} | {r['category']} | {oid} | {r.get('owasp', '—')} |")
        a("")
        a("> These attacks bypassed the guard too — the honest ceiling on this defense. Each is a")
        a("> concrete next target for a stronger or additional guard.")
    else:
        a("_No confirmed bypasses survived the guard in this run._")
    a("")

    # --- Per-category delta -------------------------------------------------------
    a("## Per-category bypass-rate delta")
    a("")
    cats = sorted({r["category"] for r in raw_rows + guarded_rows
                   if r["objective_kind"] in ("forbidden", "prompt-leak")})
    a("| Category | Raw | Guarded | Δ pts |")
    a("|---|---:|---:|---:|")
    for cat in cats:
        rk = [r for r in raw_rows if r["category"] == cat and r["objective_kind"] in ("forbidden", "prompt-leak")]
        gk = [r for r in guarded_rows if r["category"] == cat and r["objective_kind"] in ("forbidden", "prompt-leak")]
        rb = {"hits": sum(1 for r in rk if verdict_of(r) == BYPASSED and verdict_of(r) != ERROR),
              "total": sum(1 for r in rk if verdict_of(r) != ERROR)}
        gb = {"hits": sum(1 for r in gk if verdict_of(r) == BYPASSED and verdict_of(r) != ERROR),
              "total": sum(1 for r in gk if verdict_of(r) != ERROR)}
        rb["rate"] = bypass_rate(rb["hits"], rb["total"])
        gb["rate"] = bypass_rate(gb["hits"], gb["total"])
        d = (gb["rate"] - rb["rate"]) * 100.0
        a(f"| {cat} | {pct(rb['rate'])} ({rb['hits']}/{rb['total']}) | "
          f"{pct(gb['rate'])} ({gb['hits']}/{gb['total']}) | {d:+.1f} |")
    a("")

    # --- Honesty caveats ----------------------------------------------------------
    a("## Method & caveats")
    a("")
    a("- **Reference defenses, not a product.** The guards are transparent baselines — a pattern-based")
    a("  input jailbreak classifier and an output-side system-prompt DLP filter. A real deployment")
    a("  would swap in trained classifiers; the `Guard` protocol makes that a drop-in.")
    a("- **One model, one run.** The delta is measured on a single local model. Bypass is probabilistic;")
    a("  treat non-overlapping-CI reductions as defensible and overlapping ones as directional. The")
    a("  intervals capture sampling *within* this one run — not run-to-run instability, which a single")
    a("  run cannot estimate.")
    a("- **The DLP guard is handed the protected system prompt** (a realistic DLP assumption — a data-loss")
    a("  filter knows what it protects). It is never told the canary specifically; it blocks distinctive")
    a("  tokens or verbatim spans of the prompt it was given.")
    a("- **Guards are bypassable.** Every residual bypass above got through the guard; the delta is a")
    a("  measured reduction, never a claim of security.")
    a("")

    return "\n".join(lines) + "\n"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_delta_report(raw_rows: list[dict], guarded_rows: list[dict],
                       reports_dir: Path | None = None) -> Path:
    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"delta_{_stamp()}.md"
    out_path.write_text(build_delta_report(raw_rows, guarded_rows))
    return out_path
