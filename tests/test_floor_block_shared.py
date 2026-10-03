"""One capability-floor rule and one surface run sequence.

The dead / liveness-only / live rule was copied into the armed reports, disclosure, misinformation
and compare, and campaign and the CLI each wrote the load -> smoke -> chat_fn -> suite sequence.
The byte-identity goldens prove the shared versions render the same reports; these pin the rule's
boundaries and that both callers go through the one run sequence."""

from __future__ import annotations

import pytest

from iago import campaign, floor
from iago.campaign import SURFACES, SurfaceSpec
from iago.config import GROUNDING_FLOOR_MIN_CORRECT as N

from test_cli_surface_stdout import RUN_ARGV, Harness, invoke


@pytest.mark.parametrize("fired,state", [(0, "dead"), (1, "thin"), (N - 1, "thin"), (N, "live"),
                                         (N + 5, "live")])
def test_a_floor_is_dead_below_one_fire_and_live_from_the_threshold(fired, state):
    assert floor.floor_state(fired) == state


def test_compare_reads_the_same_rule():
    from iago.compare import ModelStats

    for fired in (0, 1, N - 1, N):
        m = ModelStats(model="m", floor_fired=fired, floor_total=N + 1)
        assert (m.floor_alive, m.floor_thin) == (floor.floor_state(fired) != "dead",
                                                 floor.floor_state(fired) == "thin")


def test_no_liveness_footer_without_a_thin_floor():
    from iago.arm_report import ARM_WORDING

    assert floor.liveness_footer(set(), ARM_WORDING) == []
    assert floor.liveness_footer({"b", "a"}, ARM_WORDING)[0].startswith("_⚠️ A liveness-only floor (a, b)")


class _Recorder:
    def __init__(self, monkeypatch):
        self.calls = []
        real = SurfaceSpec.run

        def run(spec, model, scenarios, **kw):
            self.calls.append((spec.key, model, [s.id for s in scenarios]))
            return real(spec, model, scenarios, **kw)
        monkeypatch.setattr(SurfaceSpec, "run", run)


def test_every_run_command_goes_through_the_surface_run_sequence(monkeypatch):
    rec = _Recorder(monkeypatch)
    for key, spec in SURFACES.items():
        Harness(monkeypatch, key)
        rc, _, _ = invoke([spec.command, *RUN_ARGV, "--smoke"])
        assert rc == 0, key
    assert [c[0] for c in rec.calls] == list(SURFACES)


def test_campaign_goes_through_the_surface_run_sequence(monkeypatch):
    rec = _Recorder(monkeypatch)
    Harness(monkeypatch, "privilege")
    _, errors = campaign.run_campaign(["privilege"], ["m1", "m2"], smoke=True, progress=False)
    assert not errors and [c[:2] for c in rec.calls] == [("privilege", "m1"), ("privilege", "m2")]
