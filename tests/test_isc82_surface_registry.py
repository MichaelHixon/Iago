"""ISC-82: one agentic surface registry, one run handler, one floor-keeping smoke slicer.

`--smoke` on every `*-run` subcommand used to take a bare `scens[:1]` (misinfo alone kept a
capability scenario). Every shipped library opens with an attack, so that slice dropped the
capability floor and a smoke run printed an uncalibrated 0% as a pass. Now every surface goes
through `campaign._smoke_slice`: first attack + first capability scenario.
"""

from __future__ import annotations

import pytest

from iago import campaign, cli
from iago.campaign import DEFAULT_SURFACES, SURFACE_REGISTRY, SURFACES

from test_cli_surface_stdout import RUN_ARGV, SURFACES as PINNED, Harness, _scen, invoke


def test_one_registry_names_every_agentic_surface_and_its_commands():
    assert list(SURFACES) == list(PINNED)
    for key, (run_cmd, scen_cmd, *_rest) in PINNED.items():
        assert (SURFACES[key].command, SURFACES[key].scenarios_command) == (run_cmd, scen_cmd)


def test_campaign_view_and_default_surfaces_are_unchanged():
    assert DEFAULT_SURFACES == ["privilege", "toolabuse", "disclosure", "misinfo"]
    for key, spec in SURFACE_REGISTRY.items():
        assert spec is SURFACES[key]   # a view of the one registry, not a second copy


@pytest.mark.parametrize("key", list(SURFACES))
def test_every_entry_resolves_its_own_modules(key):
    _, _, scen_mod, scen_attr, suite_mod, run_attr, load_attr, write_attr = PINNED[key]
    entry = SURFACES[key].entry()
    for fn, mod, attr in ((entry.load_scenarios, scen_mod, scen_attr),
                          (entry.run_suite, suite_mod, run_attr),
                          (entry.load_artifacts, suite_mod, load_attr),
                          (entry.write_report, suite_mod, write_attr)):
        module = __import__(mod, fromlist=[attr])
        assert fn is getattr(module, attr), f"{key}: {attr}"


@pytest.mark.parametrize("key", list(SURFACES))
def test_smoke_run_keeps_the_capability_floor(monkeypatch, key):
    h = Harness(monkeypatch, key)
    rc, out, _ = invoke([SURFACES[key].command, *RUN_ARGV, "--smoke"])
    assert rc == 0
    sent = h.calls[0]["scenarios"]
    assert [s.kind for s in sent] == ["attack", "capability"], [s.id for s in sent]
    assert [s.id for s in sent] == ["atk-one", "cap-one"]
    assert "  scenarios=2 trials/scenario=1 max_steps=5\n" in out


@pytest.mark.parametrize("key", list(SURFACES))
def test_smoke_slice_of_the_shipped_library_keeps_a_capability_scenario(key):
    scens = SURFACES[key].entry().load_scenarios()
    picked = campaign._smoke_slice(scens)
    # Unconditional: every shipped surface carries a capability floor, and a library that lost
    # its floor is exactly the degenerate compare the slicer exists to prevent.
    assert any(s.kind == "capability" for s in scens), f"{key}: shipped library has no capability floor"
    assert any(s.kind == "capability" for s in picked), key
    assert any(s.kind == "attack" for s in picked), key


def test_smoke_falls_back_to_the_first_scenario_without_a_capability_kind(monkeypatch):
    h = Harness(monkeypatch, "rag")
    h.scenarios = [_scen("ctl-a", "control"), _scen("ctl-b", "control")]
    rc, _, _ = invoke(["rag-run", "--smoke"])
    assert rc == 0 and [s.id for s in h.calls[0]["scenarios"]] == ["ctl-a"]


def test_no_per_surface_handler_survives():
    leftovers = [n for n in vars(cli) if n.startswith("_cmd_") and n.endswith(("_run", "_scenarios"))
                 and n not in {"_cmd_run", "_cmd_surface_run", "_cmd_surface_scenarios",
                               "_cmd_adaptive_run"}]
    assert leftovers == []
