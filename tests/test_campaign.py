"""ISC-30: cross-surface multi-model differential campaign (`iago campaign`). The pure core
rolls per-surface `compare` results into ONE consolidated report. Same honesty rule carries:
a floor-dead model is degenerate, excluded from a surface's findings — never a safety win."""

import json
from pathlib import Path

from iago.agent_oracle import HIJACKED, RESISTED
from iago.campaign import (
    SURFACE_REGISTRY,
    _smoke_slice,
    build_campaign,
    write_campaign_report,
)


def _artifact(tmp_path: Path, name: str, model: str, *, floor_fires: bool, attacks: dict[str, str]) -> Path:
    """A minimal single-model, single-surface artifact JSONL. An alive floor fires 2 capability
    trials (>= GROUNDING_FLOOR_MIN_CORRECT = certified); a dead floor is one resisted trial."""
    if floor_fires:
        rows = [{"model": model, "kind": "capability", "scenario_id": f"cap{i}", "verdict": HIJACKED}
                for i in range(2)]
    else:
        rows = [{"model": model, "kind": "capability", "scenario_id": "cap", "verdict": RESISTED}]
    for sid, verdict in attacks.items():
        rows.append({"model": model, "kind": "attack", "scenario_id": sid, "scenario_name": sid,
                     "verdict": verdict})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_build_campaign_unions_models_across_surfaces_first_seen(tmp_path):
    # Surface s1 has A then B; s2 has B then C -> union is A, B, C in first-seen order.
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    s1b = _artifact(tmp_path, "s1b.jsonl", "B", floor_fires=True, attacks={"x": RESISTED})
    s2b = _artifact(tmp_path, "s2b.jsonl", "B", floor_fires=True, attacks={"y": HIJACKED})
    s2c = _artifact(tmp_path, "s2c.jsonl", "C", floor_fires=True, attacks={"y": RESISTED})
    camp = build_campaign({"s1": [s1a, s1b], "s2": [s2b, s2c]})
    assert camp.models == ["A", "B", "C"]
    assert [sr.key for sr in camp.surfaces] == ["s1", "s2"]


def test_executive_summary_counts_findings_and_names_widest_delta(tmp_path):
    # s1: A hijacks x (100%), B resists x (0%) -> 1 finding, widest delta = x @ 100%.
    # s2: both resist y -> 0 findings, widest delta = —.
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    s1b = _artifact(tmp_path, "s1b.jsonl", "B", floor_fires=True, attacks={"x": RESISTED})
    s2a = _artifact(tmp_path, "s2a.jsonl", "A", floor_fires=True, attacks={"y": RESISTED})
    s2b = _artifact(tmp_path, "s2b.jsonl", "B", floor_fires=True, attacks={"y": RESISTED})
    camp = build_campaign({"s1": [s1a, s1b], "s2": [s2a, s2b]}, labels={"s1": "Surf One", "s2": "Surf Two"})
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    summary = text.split("## Executive summary")[1].split("## Capability-floor")[0]
    s1_row = next(ln for ln in summary.splitlines() if ln.startswith("| Surf One "))
    s2_row = next(ln for ln in summary.splitlines() if ln.startswith("| Surf Two "))
    assert "`x` (100%)" in s1_row                            # widest delta named
    assert s1_row.split("|")[3].strip() == "1"              # 1 differential finding
    assert "—" in s2_row and s2_row.split("|")[3].strip() == "0"
    assert "**Total differential findings:** 1" in text


def test_liveness_grid_glyphs_cover_every_tier(tmp_path):
    # A certified on s1 (2/2), thin on s2 (1/1); dead model D on s1; uncal model U (no capability).
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": RESISTED})
    s1d = _artifact(tmp_path, "s1d.jsonl", "D", floor_fires=False, attacks={"x": RESISTED})
    s2a = tmp_path / "s2a.jsonl"
    s2a.write_text("\n".join(json.dumps(r) for r in [
        {"model": "A", "kind": "capability", "scenario_id": "cap", "verdict": HIJACKED},  # 1/1 -> thin
        {"model": "A", "kind": "attack", "scenario_id": "y", "scenario_name": "y", "verdict": RESISTED},
    ]))
    s2u = tmp_path / "s2u.jsonl"  # U ran no capability scenario -> uncalibrated
    s2u.write_text(json.dumps(
        {"model": "U", "kind": "attack", "scenario_id": "y", "scenario_name": "y", "verdict": RESISTED}))
    camp = build_campaign({"s1": [s1a, s1d], "s2": [s2a, s2u]})
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    grid = text.split("## Capability-floor liveness")[1].split("## Per-surface")[0]
    a_row = next(ln for ln in grid.splitlines() if ln.startswith("| `A` "))
    d_row = next(ln for ln in grid.splitlines() if ln.startswith("| `D` "))
    u_row = next(ln for ln in grid.splitlines() if ln.startswith("| `U` "))
    assert a_row.split("|")[2].strip() == "✅" and a_row.split("|")[3].strip() == "✅*"
    assert d_row.split("|")[2].strip() == "❌" and d_row.split("|")[3].strip() == "–"  # D absent on s2
    assert u_row.split("|")[3].strip() == "uncal."


def test_dead_floor_model_is_excluded_from_a_surface_finding(tmp_path):
    # On s1, only A is alive-and-hijacking; D's 0% is degenerate (dead floor) -> no divergence
    # (needs >= 2 alive models to disagree). The per-surface section says so, never a finding.
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    s1d = _artifact(tmp_path, "s1d.jsonl", "D", floor_fires=False, attacks={"x": RESISTED})
    camp = build_campaign({"s1": [s1a, s1d]}, labels={"s1": "Surf One"})
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    detail = text.split("### Surf One")[1]
    assert "No divergence among calibrated models" in detail
    assert "**Total differential findings:** 0" in text


def test_per_surface_table_lists_only_alive_models_and_marks_thin(tmp_path):
    # thinA (1/1) hijacks x; certB (2/2) resists x -> a finding; thinA column starred, certB not.
    tA = tmp_path / "tA.jsonl"
    tA.write_text("\n".join(json.dumps(r) for r in [
        {"model": "thinA", "kind": "capability", "scenario_id": "cap", "verdict": HIJACKED},
        {"model": "thinA", "kind": "attack", "scenario_id": "x", "scenario_name": "x", "verdict": HIJACKED},
    ]))
    cB = _artifact(tmp_path, "cB.jsonl", "certB", floor_fires=True, attacks={"x": RESISTED})
    camp = build_campaign({"s1": [tA, cB]})
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    hdr = next(ln for ln in text.splitlines() if ln.startswith("| Scenario | Spread |"))
    assert "`thinA`*" in hdr and "`certB`*" not in hdr
    assert "liveness-only floor" in text


def test_smoke_slice_keeps_a_capability_scenario():
    # A bare [:1] would drop the floor if the first scenario is an attack; _smoke_slice keeps both.
    class S:
        def __init__(self, kind):
            self.kind = kind

    scens = [S("attack"), S("attack"), S("capability"), S("control")]
    picked = _smoke_slice(scens)
    kinds = [s.kind for s in picked]
    assert "attack" in kinds and "capability" in kinds and len(picked) == 2


def test_run_failures_are_surfaced_in_the_report_not_erased(tmp_path):
    # code-review MAJOR: a crashed run must live IN the artifact. modelB ran on s2 but FAILED on s1
    # (present in the union via s2 -> its s1 cell must read as a crash, not a `–` skip); modelC failed
    # on EVERY surface (absent from the union entirely). The report must (1) banner the failures,
    # (2) flag C as a requested-but-absent model, (3) NOT call s1 a clean negative, (4) mark B's s1
    # cell as a crash.
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    s2a = _artifact(tmp_path, "s2a.jsonl", "A", floor_fires=True, attacks={"y": RESISTED})
    s2b = _artifact(tmp_path, "s2b.jsonl", "B", floor_fires=True, attacks={"y": RESISTED})
    camp = build_campaign(
        {"s1": [s1a], "s2": [s2a, s2b]}, labels={"s1": "Surf One", "s2": "Surf Two"},
        requested_models=["A", "B", "C"],
        errors=["s1/B: ollama connection refused", "s1/C: boom", "s2/C: boom"])
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    assert "INCOMPLETE" in text and "run(s) failed" in text
    assert "s1/B: ollama connection refused" in text
    assert "absent from the entire campaign" in text and "`C`" in text  # C fully failed
    detail = text.split("### Surf One")[1].split("### Surf Two")[0]
    assert "NOT a clean negative" in detail                  # s1 had a failure -> not a clean negative
    grid = text.split("## Capability-floor liveness")[1].split("## Per-surface")[0]
    b_row = next(ln for ln in grid.splitlines() if ln.startswith("| `B` "))
    assert "⚠️ fail" in b_row                                 # B's s1 crash, distinct from a `–` skip


def test_clean_campaign_has_no_incomplete_banner(tmp_path):
    # No errors, no missing models -> the INCOMPLETE banner must NOT appear (no false alarm).
    s1a = _artifact(tmp_path, "s1a.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    s1b = _artifact(tmp_path, "s1b.jsonl", "B", floor_fires=True, attacks={"x": RESISTED})
    camp = build_campaign({"s1": [s1a, s1b]}, requested_models=["A", "B"], errors=[])
    text = write_campaign_report(camp, reports_dir=tmp_path).read_text()
    assert "INCOMPLETE" not in text


def test_registry_covers_the_four_mature_surfaces():
    assert set(SURFACE_REGISTRY) == {"privilege", "toolabuse", "disclosure", "misinfo"}
    for spec in SURFACE_REGISTRY.values():
        run_suite, load_scenarios = spec.load()   # lazy import resolves without error
        assert callable(run_suite) and callable(load_scenarios)


def test_report_carries_not_a_ranking_copy_on_findings_and_nulls(tmp_path):
    """Council (Okonkwo-Bell): the not-a-ranking + underpowered-null caption must ride INLINE
    with the numbers — under the executive summary, under a delta table, AND in a 0-finding
    result — not only in the philosophy header a screenshot of the table would crop out."""
    # A finding surface (A hijacks x, B resists) + a null surface (both resist y).
    fa = _artifact(tmp_path, "fa.jsonl", "A", floor_fires=True, attacks={"x": HIJACKED})
    fb = _artifact(tmp_path, "fb.jsonl", "B", floor_fires=True, attacks={"x": RESISTED})
    na = _artifact(tmp_path, "na.jsonl", "A", floor_fires=True, attacks={"y": RESISTED})
    nb = _artifact(tmp_path, "nb.jsonl", "B", floor_fires=True, attacks={"y": RESISTED})
    camp = build_campaign({"find": [fa, fb], "null": [na, nb]})
    out = write_campaign_report(camp, reports_dir=tmp_path)
    body = out.read_text()
    # The caption appears more than once (exec summary + the finding table), and the
    # underpowered-null truth is stated where a reader could misread a null as safety.
    assert body.count("not** a safety ranking") >= 2
    assert "underpowered" in body
    assert "absence of evidence" in body


def test_cmd_campaign_prefixes_requested_models_with_ollama(monkeypatch, tmp_path):
    """CLI-seam regression (the live-verify bug): `run_campaign` stamps artifact model
    names as `ollama:<tag>`, so the absent-model check must compare against that SAME
    canonical form. The CLI once passed bare names -> every present model read 'absent' ->
    a bogus INCOMPLETE banner. Pin that `_cmd_campaign` prefixes before `build_campaign`."""
    import types
    from iago import campaign as campaign_mod
    from iago.cli import _cmd_campaign

    captured = {}

    def fake_run_campaign(surfaces, models, **kwargs):
        return {"privilege": [tmp_path / "a.jsonl"]}, []

    def fake_build_campaign(surface_paths, *, labels, requested_models, errors):
        captured["requested_models"] = requested_models
        return object()

    monkeypatch.setattr(campaign_mod, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign_mod, "build_campaign", fake_build_campaign)
    monkeypatch.setattr(campaign_mod, "write_campaign_report", lambda camp: tmp_path / "r.md")

    args = types.SimpleNamespace(
        models="llama3.1,llama3.2:3b", surfaces="privilege", smoke=False,
        trials=1, temperature=None, base_seed=None, max_steps=None,
    )
    rc = _cmd_campaign(args)
    assert rc == 0
    assert captured["requested_models"] == ["ollama:llama3.1", "ollama:llama3.2:3b"]
