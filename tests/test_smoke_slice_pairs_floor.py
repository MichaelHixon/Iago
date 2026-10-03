"""The smoke slice pairs its attack with a capability scenario that calibrates it.

Privilege and toolabuse calibrate each arm on its own floor, disclosure each channel, misinfo each
id kind. A smoke run that sends an `rce` attack with an `ssrf` capability scenario prints the rce
row as uncalibrated, so the slicer must match the pair on the surface's floor key. The shipped
libraries happen to list a same-arm capability first; these tests reorder them so that luck is
not what passes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iago import campaign
from iago.campaign import SURFACES, smoke_slice

from test_cli_surface_stdout import RUN_ARGV, Harness, invoke

FLOOR_KEYS = {"privilege": "arm", "toolabuse": "arm", "disclosure": "channel", "misinfo": "id_kind"}


def _s(sid, kind, **floor):
    return SimpleNamespace(id=sid, kind=kind, name=sid, **floor)


def test_every_per_floor_surface_declares_its_floor_key():
    assert {k: s.floor_key for k, s in SURFACES.items() if s.floor_key} == FLOOR_KEYS


REORDERED = [_s("atk-rce", "attack", arm="rce"), _s("cap-ssrf", "capability", arm="ssrf"),
             _s("cap-rce", "capability", arm="rce")]


def test_the_capability_of_the_attacks_arm_is_picked_over_an_earlier_one_of_another_arm():
    assert [s.id for s in smoke_slice(REORDERED, "arm")] == ["atk-rce", "cap-rce"]


def test_an_attack_with_no_floor_of_its_own_gives_way_to_one_that_has_a_floor():
    """misinfo ships doi attacks with no doi capability scenario."""
    scens = [_s("fab-doi", "attack", id_kind="doi"), _s("fab-cve", "attack", id_kind="cve"),
             _s("cap-cve", "capability", id_kind="cve")]
    assert [s.id for s in smoke_slice(scens, "id_kind")] == ["fab-cve", "cap-cve"]


def test_with_no_matching_pair_it_still_keeps_a_floor():
    scens = [_s("atk-a", "attack", arm="a"), _s("cap-b", "capability", arm="b")]
    assert [s.id for s in smoke_slice(scens, "arm")] == ["atk-a", "cap-b"]


@pytest.mark.parametrize("key", sorted(FLOOR_KEYS))
def test_the_shipped_library_reversed_still_yields_a_matched_pair(key):
    spec = SURFACES[key]
    attack, cap = smoke_slice(list(reversed(spec.entry().load_scenarios())), spec.floor_key)
    assert (attack.kind, cap.kind) == ("attack", "capability")
    assert getattr(attack, spec.floor_key) == getattr(cap, spec.floor_key), (attack.id, cap.id)


def _cli_smoke():
    rc, _, _ = invoke(["tool-abuse-run", *RUN_ARGV, "--smoke"])
    assert rc == 0


def _campaign_smoke():
    _, errors = campaign.run_campaign(["toolabuse"], ["m"], smoke=True, progress=False)
    assert not errors


@pytest.mark.parametrize("smoke_run", [_cli_smoke, _campaign_smoke], ids=["cli", "campaign"])
def test_a_smoke_run_sends_the_matched_pair(monkeypatch, smoke_run):
    h = Harness(monkeypatch, "toolabuse")
    h.scenarios = REORDERED
    smoke_run()
    assert [s.id for s in h.calls[0]["scenarios"]] == ["atk-rce", "cap-rce"]
