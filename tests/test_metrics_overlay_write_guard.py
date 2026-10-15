"""ISC-51: `write_metrics` must not clobber the shipped public metrics file with a measurement a
clone cannot reproduce (an overlay measurement, or one on a different control set), while still
writing --no-overlay measurements of the shipped set and explicit-path writes.

The guard FAILS CLOSED: it writes to the shipped path only when the metrics positively verify
against the shipped set. Every test that exercises the shipped-path branch monkeypatches
`METRICS_PATH` to a tmp file, so a broken guard can never corrupt the committed
`iago/calibration/judge_metrics.json` as a test side effect."""
import json

import pytest

import iago.cli as cli
import iago.judge_eval as je
from iago.judge_eval import (
    CONTROL_SET,
    NO_OVERLAY_SENTINEL,
    OverlayWriteRefused,
    evaluate,
    load_control_set,
    write_metrics,
)


def _overlay_result():
    """A measurement whose set includes the withheld bodies filled in — set_variant=with-local-overlay."""
    public = load_control_set(CONTROL_SET, overlay=NO_OVERLAY_SENTINEL)
    filled = [dict(e, response="body", _overlay=True) if e["response"] is None else e for e in public]
    m = evaluate("heuristic", filled)
    assert m["set_variant"] == "with-local-overlay"
    return m


def _public_result():
    m = evaluate("heuristic", load_control_set(CONTROL_SET, overlay=NO_OVERLAY_SENTINEL))
    assert m["set_variant"] == "public"
    return m


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """A stand-in for the committed metrics file at METRICS_PATH — real logic, no real file touched."""
    p = tmp_path / "calibration" / "judge_metrics.json"
    p.parent.mkdir(parents=True)
    monkeypatch.setattr(je, "METRICS_PATH", p)
    return p


# --- the shipped path is protected --------------------------------------------------------------

def test_overlay_refused_at_shipped_path_leaves_it_untouched(shipped):
    """M1 (drop the raise) dies here; the refusal must raise AND the file must be byte-unchanged."""
    write_metrics(_public_result(), shipped)          # seed a legitimate public file
    before = shipped.read_bytes()
    with pytest.raises(OverlayWriteRefused) as ei:
        write_metrics(_overlay_result())              # default path resolves to METRICS_PATH
    assert "overlay-measured" in str(ei.value)
    assert shipped.read_bytes() == before


def test_public_result_writes_through_at_shipped_path(shipped):
    """The normal allowed write: a public result that verifies MUST write at the shipped path.
    Kills a mutant that makes the guard always-refuse."""
    out = write_metrics(_public_result())
    assert out == shipped and shipped.exists()
    data = json.loads(shipped.read_text())
    assert data and all(v.get("set_variant") == "public" for per in data.values() for v in per.values())


def test_force_flag_writes_overlay_at_shipped_path(shipped):
    """M4 (ignore `not allow_overlay`) dies here: the force flag MUST let an overlay result write."""
    write_metrics(_public_result(), shipped)
    before = shipped.read_bytes()
    out = write_metrics(_overlay_result(), allow_overlay=True)
    assert out == shipped and shipped.read_bytes() != before


def test_non_canonical_spelling_of_shipped_path_is_still_refused(shipped):
    """Why `.resolve()` is in the guard: a mutation to a bare `p == METRICS_PATH` dies here."""
    weird = shipped.parent / ".." / shipped.parent.name / shipped.name
    assert weird != shipped and weird.resolve() == shipped.resolve()
    with pytest.raises(OverlayWriteRefused):
        write_metrics(_overlay_result(), weird)


def test_wrong_set_refused_at_shipped_path_leaves_it_untouched(shipped):
    """A custom `--set` yields set_variant=='public' but a set_sha256 that differs from the shipped
    set — the same non-reproducible clobber by another route. Refused, and byte-unchanged."""
    write_metrics(_public_result(), shipped)          # seed
    before = shipped.read_bytes()
    m = _public_result()
    m["set_sha256"] = "0" * 64                          # as if measured on some other control set
    with pytest.raises(OverlayWriteRefused) as ei:
        write_metrics(m)
    assert "different control set" in str(ei.value)
    assert shipped.read_bytes() == before


def test_fails_closed_when_the_shipped_set_cannot_be_read(shipped, monkeypatch):
    """The writer must not be looser than the reader: when the shipped set can't be fingerprinted,
    even a correct public result is refused (it cannot be verified), naming the reason. Force still
    overrides. Pins the `expected is None` branch the Council found uncovered."""
    monkeypatch.setattr(je, "_shipped_public_fingerprint", lambda: None)
    write_metrics(_public_result(), shipped, allow_overlay=True)   # seed via the force path
    before = shipped.read_bytes()
    with pytest.raises(OverlayWriteRefused) as ei:
        write_metrics(_public_result())                # would verify if readable — refused because it isn't
    assert "could not be read" in str(ei.value)
    assert shipped.read_bytes() == before
    # force bypasses the unverifiable state
    write_metrics(_public_result(), allow_overlay=True)
    assert shipped.read_bytes() != before


# --- non-shipped paths are unaffected -----------------------------------------------------------

def test_overlay_result_writes_to_an_explicit_non_shipped_path(tmp_path):
    """M5 (path check always-True) dies here: an overlay result to a tmp path must succeed."""
    dest = tmp_path / "m.json"
    out = write_metrics(_overlay_result(), dest)
    assert out == dest
    data = json.loads(dest.read_text())
    assert any(v.get("set_variant") == "with-local-overlay" for per in data.values() for v in per.values())


def test_public_result_to_an_explicit_path_is_never_refused(tmp_path):
    dest = tmp_path / "public.json"
    out = write_metrics(_public_result(), dest)
    assert out == dest
    data = json.loads(dest.read_text())
    assert data and all(v.get("set_variant") == "public" for per in data.values() for v in per.values())


# --- the CLI flag actually reaches the guard ----------------------------------------------------

def test_cli_allow_overlay_write_flag_reaches_write_metrics(monkeypatch):
    """Pins the argparse flag → write_metrics(allow_overlay=...) plumbing end to end, both states."""
    captured = {}

    def fake_write(result, path=None, *, allow_overlay=False):
        captured["allow_overlay"] = allow_overlay
        return je.METRICS_PATH

    monkeypatch.setattr(je, "write_metrics", fake_write)

    assert cli.main(["judge-eval", "--judge", "heuristic", "--no-overlay", "--allow-overlay-write"]) == 0
    assert captured["allow_overlay"] is True

    captured.clear()
    assert cli.main(["judge-eval", "--judge", "heuristic", "--no-overlay"]) == 0
    assert captured["allow_overlay"] is False
