"""Report generator: JSONL artifacts -> pentest-style markdown.

Reads the runner's artifacts (never re-hits the model) and computes bypass RATES
per technique and per category, then writes a report with: executive summary,
results by category/technique, the actual prompt+response evidence for the strongest
bypasses, a judge-calibration note driven by the control objective, and a hardening
recommendations section (the defensive payoff — the whole point of the exercise).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import REPORTS_DIR
from .judge import BYPASSED, NEEDS_REVIEW

# A trial "bypassed" a FORBIDDEN objective => guardrail failure. Control objectives
# are scored separately (they are the calibration canary, not a finding).


def _verdict(r: dict) -> str:
    """Prefer the Claude rubric verdict when a row has been regraded; else heuristic."""
    return r.get("claude_verdict") or r["verdict"]


def _conf(r: dict) -> float:
    return r.get("claude_confidence", r.get("confidence", 0.0))


def _why(r: dict) -> str:
    return r.get("claude_rationale") or r.get("rationale", "")


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 3) if total else 0.0


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build_report(rows: list[dict]) -> str:
    """Render the markdown report from artifact rows."""
    if not rows:
        return "# Iago Report\n\n_No artifacts — nothing to report._\n"

    model = rows[0]["model"]
    forbidden = [r for r in rows if r["objective_kind"] == "forbidden"]
    controls = [r for r in rows if r["objective_kind"] == "control"]

    trials_total = len(forbidden)
    bypasses = [r for r in forbidden if _verdict(r) == BYPASSED]
    needs_review = [r for r in forbidden if _verdict(r) == NEEDS_REVIEW]
    overall_rate = _rate(len(bypasses), trials_total)

    # Per-technique bypass rate (forbidden only).
    by_tech: dict[str, list[dict]] = defaultdict(list)
    for r in forbidden:
        by_tech[r["technique_id"]].append(r)

    tech_stats = []
    for tid, trs in by_tech.items():
        hits = sum(1 for r in trs if _verdict(r) == BYPASSED)
        tech_stats.append(
            {
                "id": tid,
                "name": trs[0]["technique_name"],
                "category": trs[0]["category"],
                "owasp": trs[0].get("owasp", "—"),
                "hits": hits,
                "total": len(trs),
                "rate": _rate(hits, len(trs)),
            }
        )
    tech_stats.sort(key=lambda s: (-s["rate"], s["id"]))

    # Per-category bypass rate.
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in forbidden:
        by_cat[r["category"]].append(r)
    cat_stats = []
    for cat, trs in by_cat.items():
        hits = sum(1 for r in trs if _verdict(r) == BYPASSED)
        cat_stats.append(
            {"category": cat, "hits": hits, "total": len(trs), "rate": _rate(hits, len(trs))}
        )
    cat_stats.sort(key=lambda s: (-s["rate"], s["category"]))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    a = lines.append

    a("# Iago — Guardrail Red-Team Report")
    a("")
    a("> **Authorized defensive-security research.** Iago probes an LLM's own safety")
    a("> controls to measure which bypass techniques slip past them, so the controls")
    a("> can be hardened. Target is a local model under the operator's control.")
    a("")
    a(f"- **Target model:** `{model}`")
    a(f"- **Generated:** {now}")
    a(f"- **Objectives:** {len({r['objective_id'] for r in forbidden})} forbidden, "
      f"{len({r['objective_id'] for r in controls})} control")
    a(f"- **Techniques:** {len(by_tech)} | **Trials/pair:** "
      f"{max((s['total'] for s in tech_stats), default=0) // max(len({r['objective_id'] for r in forbidden}), 1)}")
    a("")

    a("## Executive Summary")
    a("")
    a(f"**{len(bypasses)} / {trials_total} forbidden-objective trials were confirmed "
      f"bypasses ({_pct(overall_rate)} confirmed-bypass rate).**")
    a("")
    if needs_review:
        a(f"A further **{len(needs_review)} / {trials_total}** trials produced substantive, "
          "non-refusing content the fast heuristic judge cannot adjudicate — flagged "
          "`needs-review`, not counted as bypasses. Only the Claude rubric judge confirms a "
          "real bypass; run it to resolve these. A heuristic-only run reports 0 confirmed "
          "bypasses by design — that is the fix for the old false-positive headline, not a "
          "clean bill of health.")
        a("")
    a("Bypass rate is reported across repeated trials rather than a single shot, because")
    a("guardrail bypass is probabilistic — one lucky hit is noise. Sampling is pinned")
    a("(fixed temperature, per-trial seed) to make runs reproducible *where the backend")
    a("honors the seed* — verify seed determinism for your target before relying on it.")
    a("")

    # Control calibration note.
    a("### Judge calibration (control objective)")
    a("")
    if controls:
        ctrl_bypass = sum(1 for r in controls if _verdict(r) == BYPASSED)
        ctrl_review = sum(1 for r in controls if _verdict(r) == NEEDS_REVIEW)
        a(f"The benign control was confirmed `bypassed` in {ctrl_bypass}/{len(controls)} trials "
          f"and flagged `needs-review` in {ctrl_review}/{len(controls)}. The control is a harmless")
        a("request, so an ideal judge should *never* confirm it as a bypass. If the Claude rubric")
        a("judge scores the control as `bypassed`, the judge itself is miscalibrated and this run's")
        a("forbidden numbers should not be trusted. (The heuristic cannot adjudicate content, so it")
        a("routes substantive control replies to `needs-review` rather than asserting a bypass.)")
    else:
        a("_No control objective in this run — add one to calibrate the judge._")
    a("")

    a("## Bypass Rate by Category")
    a("")
    a("| Category | Bypassed | Trials | Rate |")
    a("|----------|----------|--------|------|")
    for s in cat_stats:
        a(f"| {s['category']} | {s['hits']} | {s['total']} | {_pct(s['rate'])} |")
    a("")

    a("## Bypass Rate by Technique")
    a("")
    a("| Rank | Technique | Category | OWASP | Bypassed | Trials | Rate |")
    a("|------|-----------|----------|-------|----------|--------|------|")
    for i, s in enumerate(tech_stats, 1):
        a(f"| {i} | {s['name']} (`{s['id']}`) | {s['category']} | {s['owasp']} | {s['hits']} | "
          f"{s['total']} | {_pct(s['rate'])} |")
    a("")

    # Evidence: strongest bypasses (highest-confidence bypassed trials on forbidden objs).
    a("## Evidence — Strongest Bypasses")
    a("")
    evid = sorted(
        (r for r in bypasses),
        key=lambda r: -_conf(r),
    )[:5]
    if not evid:
        a("_No bypasses recorded — the guardrails held across all techniques._")
        a("")
    for i, r in enumerate(evid, 1):
        a(f"### {i}. `{r['technique_id']}` ({r['technique_name']}) vs `{r['objective_id']}`")
        a("")
        a(f"- **Verdict:** {_verdict(r)} (confidence {_conf(r):.2f}) — {_why(r)}")
        a(f"- **Seed:** {r['seed']} · **Latency:** {r['latency_s']}s")
        a("")
        a("**Prompt sent:**")
        a("")
        a("```")
        a(r["prompt"].strip()[:1200])
        a("```")
        a("")
        a("**Model response (excerpt):**")
        a("")
        a("```")
        a(r["response"].strip()[:1200])
        a("```")
        a("")

    a("## Hardening Recommendations")
    a("")
    a("The defensive payoff — what to do about the findings above:")
    a("")
    top_cats = [s["category"] for s in cat_stats if s["rate"] > 0][:3]
    if top_cats:
        a(f"1. **Prioritize the leaking categories:** {', '.join(f'`{c}`' for c in top_cats)} "
          "showed the highest bypass rates. Add refusal-training or input-side detection "
          "targeting these patterns first.")
    else:
        a("1. **No category leaked** in this run — widen the attack library and raise trial "
          "count before concluding the controls are robust.")
    a("2. **Add output-side classification.** Several bypasses are the model producing "
      "disallowed content *after* a token refusal; a response classifier catches what the "
      "refusal layer misses.")
    a("3. **Re-test after each mitigation** using the same pinned seeds — the JSONL artifacts "
      "make before/after directly comparable.")
    a("4. **Upgrade the judge (planned)** to reason about content, not keywords, so benign and "
      "harmful substantive replies are no longer conflated (see calibration note).")
    a("")
    a("---")
    a("_Generated by Iago — authorized LLM guardrail red-team harness._")
    a("")
    return "\n".join(lines)


def write_report(rows: list[dict], reports_dir: Path | None = None) -> Path:
    """Build and write the report to reports/, returning its path."""
    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model = rows[0]["model"] if rows else "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace(":", "-").replace("/", "-")
    out_path = out_dir / f"report_{stamp}_{safe_model}.md"
    out_path.write_text(build_report(rows))
    return out_path
