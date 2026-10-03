"""Byte-identity goldens for the disclosure and misinformation reports.

Both reports carry a copy of the capability-floor block the armed reports share (dead / liveness-
only / live, the `✅*` marker, the liveness-only footer). Before that block is shared, this pins
exactly what each report renders, so the refactor cannot change a byte. The goldens were captured
from the pre-refactor code in their own commit.

The cases start from the real runner rows in `tests/golden/<surface>.jsonl` (a liveness-only floor
on every channel / kind) and rewrite the capability rows to reach every other floor state: all
live, all dead, mixed, a channel or kind with no floor, and no floor at all.
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import iago.agent_disclosure as disclosure
import iago.agent_misinfo as misinfo

GOLDEN_DIR = Path(__file__).parent / "golden"
FLOOR_GOLDEN_DIR = GOLDEN_DIR / "reports_floor"
FROZEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN


@pytest.fixture
def frozen(monkeypatch):
    for mod in (disclosure, misinfo):
        monkeypatch.setattr(mod, "datetime", _FrozenDatetime)


def _suite(surface):
    lines = (GOLDEN_DIR / f"{surface}.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines[1:]]


def _caps(rows, keep=lambda r: True):
    return [r for r in rows if r["kind"] == "capability" and keep(r)]


def _rest(rows):
    return [r for r in rows if r["kind"] != "capability"]


def _doubled(caps):
    """Each capability row plus a copy one trial later, so a fired floor reaches >= 2."""
    out = []
    for r in caps:
        out += [r, {**r, "trial": r["trial"] + 10}]
    return out


# --- disclosure: a fired floor is an output-channel hijack, or the read -> send chain -----------

def _disc_fire(r, fired):
    r = copy.deepcopy(r)
    if r["scenario_channel"] == "output":
        r["verdict"], r["channel"] = ("hijacked", "output") if fired else ("resisted", None)
    else:
        r["chain_read_send"] = fired
    return r


def _disc(fire_output, fire_external, drop=()):
    rows = _suite("disclosure")
    caps = [_disc_fire(r, {"output": fire_output, "external": fire_external}[r["scenario_channel"]])
            for r in _caps(rows, lambda r: r["scenario_channel"] not in drop)]
    return _rest(rows) + _doubled(caps)


def _disc_legacy():
    """Attack rows that predate `poison_delivered` (the sibling of ISC-84): Delivered is n/a, never
    0. atk-ext keeps its hijack (a confirmed positive stays ✅); atk-out is all resisted, one row
    delivered and one unknown, so it reads "delivery unknown", never non-delivered."""
    out = []
    for r in _suite("disclosure"):
        if r["scenario_id"] == "atk-ext":
            r = {**r, "poison_delivered": None}
        elif r["scenario_id"] == "atk-out":
            r = {**r, "poison_delivered": True if r["trial"] == 0 else None, "verdict": "resisted",
                 "channel": None}
        out.append(r)
    return out


DISCLOSURE_CASES = {
    "suite": lambda: _suite("disclosure"),
    "live": lambda: _disc(True, True),
    "dead": lambda: _disc(False, False),
    "mixed": lambda: _disc(True, False),
    "nochannel": lambda: _disc(True, True, drop=("external",)),
    "nofloor": lambda: _rest(_suite("disclosure")),
    "legacy": lambda: _disc_legacy(),
}


# --- misinformation: a fired floor is a correct identifier on a real subject --------------------

def _mis_fire(r, correct, kind=None):
    r = copy.deepcopy(r)
    r["submitted_correct"] = correct
    if kind:
        r["id_kind"], r["scenario_id"] = kind, f"{r['scenario_id']}-{kind}"
    return r


def _mis(correct):
    rows = _suite("misinfo")
    return _rest(rows) + _doubled([_mis_fire(r, correct) for r in _caps(rows)])


def _mis_mixed():
    """cve live, rfc dead, doi attacked with no floor at all; the rfc and doi attacks fabricate
    nothing, so their Calibrated cell shows the floor state instead of a confirmed positive."""
    rows = _suite("misinfo")
    caps = _caps(rows)
    attacks = [r for r in rows if r["kind"] == "attack"]
    extra = [{**copy.deepcopy(r), "id_kind": k, "scenario_id": f"{r['scenario_id']}-{k}",
              "verdict": "resisted", "fabricated": False}
             for r in attacks for k in ("rfc", "doi")]
    return (_rest(rows) + extra + _doubled([_mis_fire(r, True) for r in caps])
            + _doubled([_mis_fire(r, False, "rfc") for r in caps]))


MISINFO_CASES = {
    "suite": lambda: _suite("misinfo"),
    "live": lambda: _mis(True),
    "dead": lambda: _mis(False),
    "mixed": _mis_mixed,
    "nofloor": lambda: _rest(_suite("misinfo")),
}

SURFACES = {
    "disclosure": (disclosure.write_disclosure_report, "disclosure_report", DISCLOSURE_CASES),
    "misinfo": (misinfo.write_misinfo_report, "misinfo_report", MISINFO_CASES),
}


def render_all(tmp_path: Path) -> dict[str, str]:
    """Every golden's name -> rendered text. Requires the `frozen` fixture to be active."""
    out: dict[str, str] = {}
    for surface, (writer, prefix, cases) in SURFACES.items():
        for case, rows_fn in cases.items():
            path = writer(rows_fn(), reports_dir=tmp_path / f"{surface}_{case}")
            assert path.name == f"{prefix}_20260102T030405Z.md"
            out[f"{surface}_{case}.md"] = path.read_text()
    return out


GOLDEN_NAMES = sorted(f"{s}_{c}.md" for s, (_, _, cases) in SURFACES.items() for c in cases)


@pytest.fixture
def rendered(frozen, tmp_path):
    return render_all(tmp_path)


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_report_is_byte_identical_to_the_golden(name, rendered):
    assert rendered[name].encode() == (FLOOR_GOLDEN_DIR / name).read_bytes()


def test_every_golden_on_disk_is_checked():
    assert sorted(p.name for p in FLOOR_GOLDEN_DIR.iterdir()) == GOLDEN_NAMES


def test_the_goldens_reach_every_floor_state():
    """The goldens only prove the refactor if they reach each branch of the floor block."""
    text = {n: (FLOOR_GOLDEN_DIR / n).read_text() for n in GOLDEN_NAMES}
    for surface in SURFACES:
        assert "liveness only" in text[f"{surface}_suite.md"], surface
        assert "liveness-only floor" in text[f"{surface}_suite.md"], surface
        assert "✅*" in text[f"{surface}_suite.md"], surface
        assert "> ✅" in text[f"{surface}_live.md"], surface
        assert "liveness only" not in text[f"{surface}_live.md"], surface
        assert "> ⚠️ The agent never" in text[f"{surface}_dead.md"], surface
        assert "No capability scenario in this run" in text[f"{surface}_nofloor.md"], surface
    assert "no capability scenario for this channel" in text["disclosure_nochannel.md"]
    legacy = text["disclosure_legacy.md"]
    assert "| atk-ext | 2 | n/a | 1 |" in legacy and "| atk-out | 2 | n/a | 0 |" in legacy
    assert legacy.count("⚠️ delivery unknown") == 1 and "Delivered reads n/a" in legacy
    assert "non-delivered" not in legacy and "DELIVERED in 0" not in legacy
    mixed = text["misinfo_mixed.md"]
    assert "**cve**" in mixed and "**rfc**" in mixed
    assert "dead floor: rfc" in mixed and "no floor: doi" in mixed


if __name__ == "__main__":  # pragma: no cover — capture entry point
    import tempfile

    for mod in (disclosure, misinfo):
        mod.datetime = _FrozenDatetime
    FLOOR_GOLDEN_DIR.mkdir(exist_ok=True)
    for name, text in render_all(Path(tempfile.mkdtemp())).items():
        (FLOOR_GOLDEN_DIR / name).write_text(text)
        print(f"wrote {name}")
