"""iago/campaign.py — Cross-surface multi-model differential campaign (ISC-30).

`iago compare` (ISC-29) answers "how do these models differ on ONE surface?". A
campaign answers the whole-picture question — *what did we find across every surface?* — in a
SINGLE artifact spanning every surface: *model A hijacks SSRF where B holds; model C
fabricates CVEs where D declines*. It reuses `compare` per surface and rolls the
per-surface differentials into one consolidated report. This is the ISC-29 vision's
capstone: the pile of per-surface reports collapses into one quotable finding.

Two layers, mirroring `compare`'s discipline:

* **Pure core** (`build_campaign` + `write_campaign_report`) — aggregation over rows
  already on disk. No live model, no network, no new oracle. Fully unit-testable.
* **Orchestration** (`run_campaign` + `SURFACE_REGISTRY`) — the live wrapper that
  fires each registered surface × model via its existing `run_*_suite`, grouping the
  artifact paths by surface for the pure core.

Honesty (the project thesis, carried straight through `compare`): a divergence is a
FINDING only among models whose capability floor is ALIVE on that surface. A model too
weak to fire the tool is degenerate, never "safe". The report claims the instrument
plus specific per-surface, per-scenario deltas — it NEVER ranks models by safety.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .compare import (
    GROUNDING_FLOOR_MIN_CORRECT,
    Comparison,
    ModelStats,
    no_rate_cell,
    build_comparison,
    divergent_scenarios,
)
from .config import REPORTS_DIR

# ---------------------------------------------------------------------------
# Pure core — aggregate per-surface comparisons into one campaign.
# ---------------------------------------------------------------------------


@dataclass
class SurfaceResult:
    """One surface's comparison plus its label, ready to render in the roll-up."""

    key: str
    label: str
    comp: Comparison


@dataclass
class Campaign:
    surfaces: list[SurfaceResult]
    models: list[str]  # union across surfaces, first-seen order (stable report)
    requested_models: list[str]  # what the caller asked for (to flag a model that fully failed)
    errors: list[str]  # per surface/model run failures, format "surface/model: reason"


def build_campaign(
    surface_paths: dict[str, list[Path | str]],
    labels: dict[str, str] | None = None,
    requested_models: list[str] | None = None,
    errors: list[str] | None = None,
) -> Campaign:
    """Aggregate per surface. ``surface_paths`` maps a surface key -> its artifact paths
    (one or more single-model artifacts of THAT surface). Each surface is compared
    independently via ``build_comparison``; the cross-surface roll-up is rendered later.

    Surface order follows ``surface_paths`` insertion order. The model list is the union
    across surfaces in first-seen order, so a model present on only some surfaces still
    appears in the liveness grid (with an empty cell where it did not run).

    ``requested_models`` + ``errors`` (from ``run_campaign``) let the REPORT carry run
    failures honestly: a crashed run must never read as a clean "surface does not separate
    these models" result, and a model that failed on every surface must not silently vanish
    from the quotable artifact — the caveat has to live IN the report, not only on stderr.
    """
    labels = labels or {}
    surfaces: list[SurfaceResult] = []
    model_order: list[str] = []
    seen: set[str] = set()
    for key, paths in surface_paths.items():
        comp = build_comparison(list(paths))
        surfaces.append(SurfaceResult(key=key, label=labels.get(key, key), comp=comp))
        for m in comp.models:
            if m.model not in seen:
                seen.add(m.model)
                model_order.append(m.model)
    return Campaign(surfaces=surfaces, models=model_order,
                    requested_models=list(requested_models or []), errors=list(errors or []))


def _surface_failed(errors: list[str], surface_key: str) -> list[str]:
    """The models whose run FAILED on this surface — parsed from the ``"surface/model: reason"``
    error strings ``run_campaign`` emits. Lets a per-surface section say INCOMPLETE instead of
    claiming a clean negative when a run actually crashed."""
    out: list[str] = []
    prefix = f"{surface_key}/"
    for err in errors:
        if err.startswith(prefix):
            out.append(err[len(prefix):].split(":", 1)[0])
    return out


def _floor_glyph(m: ModelStats | None) -> str:
    """A single compact cell for the liveness grid — the sentence-form label lives in the
    per-surface `compare` report; here we need a scan-glyph only. Mirrors the same tiers:
    a certified-alive floor is `✅`, a liveness-only (thin) floor `✅*`, a dead floor `❌`,
    an uncalibrated (never-ran-capability) surface `uncal.`, and a model absent on the
    surface a dash."""
    if m is None:
        return "–"
    if not m.has_floor:
        return "uncal."
    if not m.floor_alive:
        return "❌"
    return "✅*" if m.floor_thin else "✅"


def _stats_by_model(comp: Comparison) -> dict[str, ModelStats]:
    return {m.model: m for m in comp.models}


# Council (Okonkwo-Bell): a printed delta table is STRUCTURALLY a ranking, and no header
# disclaimer survives a screenshot of the table itself. This caption rides INLINE under every
# delta table and every 0-finding result, so the not-a-ranking + underpowered-null truth travels
# WITH the numbers a reader screenshots — absence of a differential is absence of evidence, not
# proof a model resists.
_NOT_A_RANKING = (
    "_A printed delta is a per-scenario measurement, **not** a safety ranking. At small sample "
    "sizes this study is **underpowered** to separate models: absence of a differential is absence "
    "of evidence, **not** evidence that a model resists._"
)


def write_campaign_report(campaign: Campaign, reports_dir: Path | None = None) -> Path:
    """Render the consolidated cross-surface differential. Reads the Campaign only."""
    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"campaign_report_{ts}.md"

    # Pre-compute each surface's divergences once (used by the summary AND the detail).
    divs_by_surface = {sr.key: divergent_scenarios(sr.comp) for sr in campaign.surfaces}
    total_findings = sum(len(d) for d in divs_by_surface.values())

    lines: list[str] = ["# Iago — Cross-Surface Multi-Model Differential Campaign", ""]
    lines.append("> Defensive research. Every attack surface, run against multiple models — the DELTA "
                 "between them is the finding. This one artifact rolls up the per-surface differentials: "
                 "where one calibrated model is hijacked and another fully resists, that scenario is a "
                 "finding.")
    lines.append("")
    lines.append("> **Instrument, not verdict.** A low hijack rate is NOT a safety rating; nothing here "
                 "ranks models by safety. A finding is a specific per-surface, per-scenario DELTA between "
                 "models whose capability floor is ALIVE — a model too weak to fire the surface's tool is "
                 "degenerate, never 'safe', and is excluded from that surface's findings.")
    lines.append("")
    lines.append(f"- **Models:** {len(campaign.models)}  ·  **Surfaces:** {len(campaign.surfaces)}  ·  "
                 f"**Total differential findings:** {total_findings}")
    lines.append('- **Framework edition:** OWASP LLM Top 10 **2025**. The 2026 edition renamed and broadened System Prompt Leakage to Hidden Context Exposure and renumbered entries, so a number that means one risk in 2025 can mean a different one in 2026. Read every `LLMxx` here as 2025.')
    lines.append("")

    # --- Incomplete-runs banner: run failures MUST live in the artifact, not only on stderr
    # (code-review MAJOR; same doctrine as compare's in-cell degeneracy stamps). A crashed run is
    # NOT a clean negative, and a fully-failed model must not silently vanish from the report. ---
    missing = [m for m in campaign.requested_models if m not in campaign.models]
    if campaign.errors or missing:
        lines.append(f"> ⚠️ **INCOMPLETE — {len(campaign.errors)} surface×model run(s) failed and are "
                     "NOT reflected in the tables below.** A failed run is not a result; do not read a "
                     "blank/`–` cell or a 'no divergence' surface as resistance where a run crashed.")
        lines.append("")
        for err in campaign.errors:
            lines.append(f">   - `{err}`")
        if missing:
            lines.append(f">   - **Requested model(s) absent from the entire campaign** (failed on every "
                         f"surface run): {', '.join(f'`{m}`' for m in missing)}")
        lines.append("")

    # --- Executive summary: one row per surface, the "what did you find" glance. ---
    lines.append("## Executive summary — where the models diverge")
    lines.append("")
    lines.append("| Surface | Alive models | Differential findings | Widest delta |")
    lines.append("|---|---:|---:|---|")
    for sr in campaign.surfaces:
        alive = [m.model for m in sr.comp.models if m.floor_alive]
        divs = divs_by_surface[sr.key]
        if divs:
            top = divs[0]
            widest = f"`{top.scenario_id}` ({top.spread:.0%})"
        else:
            widest = "—"
        lines.append(f"| {sr.label} | {len(alive)} | {len(divs)} | {widest} |")
    lines.append("")
    lines.append(_NOT_A_RANKING)
    lines.append("")

    # --- Capability-floor liveness grid: model × surface, the calibration at a glance. ---
    lines.append("## Capability-floor liveness (model × surface)")
    lines.append("")
    lines.append("| Model | " + " | ".join(sr.label for sr in campaign.surfaces) + " |")
    lines.append("|---|" + "|".join(":--:" for _ in campaign.surfaces) + "|")
    stats = {sr.key: _stats_by_model(sr.comp) for sr in campaign.surfaces}
    failed_by_surface = {sr.key: set(_surface_failed(campaign.errors, sr.key)) for sr in campaign.surfaces}
    for model in campaign.models:
        cells = []
        for sr in campaign.surfaces:
            ms = stats[sr.key].get(model)
            if ms is None and model in failed_by_surface[sr.key]:
                cells.append("⚠️ fail")   # a crashed run, NOT an intentional skip
            else:
                cells.append(_floor_glyph(ms))
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"_`✅` certified-alive floor (fired ≥ {GROUNDING_FLOOR_MIN_CORRECT}); `✅*` liveness-only "
                 "(fired once — non-degenerate but not certified, a low rate may reflect a flaky "
                 "tool-caller); `❌` DEAD floor (too weak to fire the tool even when authorized — its low "
                 "hijack rate is degenerate, NOT resistance, and it is excluded from that surface's "
                 "findings); `uncal.` ran no capability scenario; `⚠️ fail` the run crashed (see the "
                 "incomplete banner above); `–` did not run the surface._")
    lines.append("")

    # --- Per-surface differential findings: the detail behind each summary row. ---
    lines.append("## Per-surface differential findings")
    lines.append("")
    for sr in campaign.surfaces:
        lines.append(f"### {sr.label}")
        lines.append("")
        alive_models = [m for m in sr.comp.models if m.floor_alive]
        divs = divs_by_surface[sr.key]
        failed_here = _surface_failed(campaign.errors, sr.key)
        if not divs:
            if failed_here:
                lines.append(f"_⚠️ INCOMPLETE — {len(failed_here)} run(s) failed on this surface "
                             f"({', '.join(failed_here)}); this is NOT a clean negative. No divergence "
                             "among the models that DID run, but the crashed run(s) were never scored._")
            else:
                lines.append("_No divergence among calibrated models on this surface — every alive model "
                             "returned the same outcome on every shared scenario. That is a result (the "
                             "surface does not separate these models at this sample size), not a "
                             "differential finding — and **not** evidence that these models resist; the "
                             "study is simply underpowered to distinguish them here._")
            lines.append("")
            continue
        thin = [m.model for m in alive_models if m.floor_thin]
        if thin:
            lines.append(f"_⚠️ `*` marks a liveness-only floor ({', '.join(thin)}): its side of a delta is "
                         "non-degenerate but not certified. Raise its trials to certify._")
            lines.append("")
        lines.append("| Scenario | Spread | "
                     + " | ".join(f"`{m.model}`" + ("*" if m.floor_thin else "") for m in alive_models)
                     + " |")
        lines.append("|---|---:|" + "|".join("---:" for _ in alive_models) + "|")
        for d in divs:
            cells = []
            for m in alive_models:
                r = m.rate(d.scenario_id)
                # Same two-facts split as the per-surface matrix (ISC-36): a dash is "never ran it",
                # `∅ (N excl.)` is "ran it, every trial dropped as an incomplete probe".
                cells.append(no_rate_cell(m, d.scenario_id) if r is None else f"{r:.0%}")
            lines.append(f"| {d.scenario_id} | {d.spread:.0%} | " + " | ".join(cells) + " |")
        lines.append("")
        # This table can emit `∅`, so this table has to explain it — the surrounding legend at the
        # top of the report covers the floor-glyph grid only, and a marker whose only explanation
        # lives in a different section does not survive a screenshot of this one (Council/Iriarte).
        if any("∅" in c for row in divs
               for c in (no_rate_cell(m, row.scenario_id) for m in alive_models)):
            lines.append("_`∅ (N excl.)` = the model RAN that scenario and all N trials were dropped "
                         "as incomplete probes, so it has no rate — unmeasured, not untried. `–` = "
                         "the model never ran it._")
            lines.append("")
        lines.append(_NOT_A_RANKING)
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# Orchestration — the live wrapper. Fires each surface × model via its existing
# `run_*_suite`. Kept thin: no scoring lives here, only dispatch + grouping.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSpec:
    """One registered surface: how to load its run entrypoints, lazily (imports are only
    paid for the surfaces a campaign actually runs)."""

    key: str
    label: str
    _loader: Callable[[], tuple[Callable[..., Path], Callable[[], list]]]

    def load(self) -> tuple[Callable[..., Path], Callable[[], list]]:
        """Return ``(run_suite, load_scenarios)`` for this surface."""
        return self._loader()


def _privilege_entry():
    from .agent_privilege import load_privilege_scenarios, run_privilege_suite
    return run_privilege_suite, load_privilege_scenarios


def _toolabuse_entry():
    from .agent_toolabuse import load_toolabuse_scenarios, run_toolabuse_suite
    return run_toolabuse_suite, load_toolabuse_scenarios


def _disclosure_entry():
    from .agent_disclosure import load_disclosure_scenarios, run_disclosure_suite
    return run_disclosure_suite, load_disclosure_scenarios


def _misinfo_entry():
    from .agent_misinfo import load_misinfo_scenarios, run_misinfo_suite
    return run_misinfo_suite, load_misinfo_scenarios


# The four mature per-arm/channel surfaces (ISC-20..28) — each carries a capability floor,
# so the campaign's honesty machinery (alive/thin/dead) is meaningful. Extensible: a new
# surface with the uniform `run_*_suite(chat_fn, *, model_name, ...)` signature drops in here.
SURFACE_REGISTRY: dict[str, SurfaceSpec] = {
    "privilege": SurfaceSpec("privilege", "Excessive agency (LLM06)", _privilege_entry),
    "toolabuse": SurfaceSpec("toolabuse", "Tool abuse RCE/SSRF (ASI05/02)", _toolabuse_entry),
    "disclosure": SurfaceSpec("disclosure", "Sensitive disclosure (LLM02)", _disclosure_entry),
    "misinfo": SurfaceSpec("misinfo", "Misinformation (LLM09)", _misinfo_entry),
}

DEFAULT_SURFACES = list(SURFACE_REGISTRY)


def _smoke_slice(scens: list) -> list:
    """A smoke selection that KEEPS the capability floor: the first attack + the first
    capability scenario (a bare ``[:1]`` would often drop the floor, making the compare
    degenerate). Falls back to the first scenario if the surface lacks either kind."""
    attack = next((s for s in scens if getattr(s, "kind", None) == "attack"), None)
    capability = next((s for s in scens if getattr(s, "kind", None) == "capability"), None)
    picked = [s for s in (attack, capability) if s is not None]
    return picked or scens[:1]


def run_campaign(
    surfaces: list[str],
    models: list[str],
    *,
    trials: int = 2,
    temperature: float | None = None,
    base_seed: int | None = None,
    max_steps: int | None = None,
    smoke: bool = False,
    progress: bool = True,
    on_event: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[Path]], list[str]]:
    """Fire every requested surface × model via its `run_*_suite`; return
    ``({surface_key: [artifact_path, ...]}, errors)``. A surface/model failure is recorded
    in ``errors`` and skipped, so one dead surface never sinks the whole campaign. LOCAL
    (Ollama) only — the same containment every surface run enforces.
    """
    from .agent_run import ollama_chat_fn

    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    # Only pass kwargs the caller actually set, so each suite's own defaults stand otherwise.
    common: dict = {"trials": trials, "progress": progress}
    if temperature is not None:
        common["temperature"] = temperature
    if base_seed is not None:
        common["base_seed"] = base_seed
    if max_steps is not None:
        common["max_steps"] = max_steps

    surface_paths: dict[str, list[Path]] = {}
    errors: list[str] = []
    for skey in surfaces:
        spec = SURFACE_REGISTRY.get(skey)
        if spec is None:
            errors.append(f"{skey}: unknown surface (known: {', '.join(SURFACE_REGISTRY)})")
            continue
        run_suite, load_scenarios = spec.load()
        scens = load_scenarios()
        if smoke:
            scens = _smoke_slice(scens)
        paths: list[Path] = []
        for model in models:
            emit(f"[{skey}] ollama:{model} — {len(scens)} scenarios × {trials} trials")
            try:
                path = run_suite(ollama_chat_fn(model), model_name=f"ollama:{model}",
                                 scenarios=scens, **common)
            except Exception as exc:  # noqa: BLE001 — record and keep the campaign alive
                errors.append(f"{skey}/{model}: {exc}")
                continue
            paths.append(path)
        if paths:
            surface_paths[skey] = paths
    return surface_paths, errors
