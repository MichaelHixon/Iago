"""The capability-floor block shared by the per-arm / per-channel / per-kind agentic reports.

Privilege and toolabuse (`arm_report`), disclosure and misinformation each calibrate an attack
rate against a capability floor of its own arm, channel or id kind. A floor is DEAD when it never
fired (the attack rate is degenerate), LIVENESS-ONLY when it fired fewer than
`GROUNDING_FLOOR_MIN_CORRECT` times (the path works, but that is not reliable capability, so the
table marks the row `✅*`), and LIVE otherwise. The state rule and the liveness-only prose live
here once; each surface supplies only its nouns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import GROUNDING_FLOOR_MIN_CORRECT

FloorState = Literal["dead", "thin", "live"]


def floor_state(fired: int) -> FloorState:
    """"dead" (never fired), "thin" (fired, but fewer than `GROUNDING_FLOOR_MIN_CORRECT` times: a
    liveness proxy, not reliable capability), or "live"."""
    if fired == 0:
        return "dead"
    return "thin" if fired < GROUNDING_FLOOR_MIN_CORRECT else "live"


@dataclass(frozen=True)
class FloorWording:
    """The nouns one surface's floor block is written in."""
    unit: str                    # what a floor is per: "arm", "channel", "kind"
    path: str                    # what FIRES: "path", "channel"
    fire_nouns: tuple[str, str]  # (singular, plural) for the fire count: ("fire", "fires")
    rate: str                    # the attack rate the floor calibrates, with `{label}` in it
    calibrated_for: str          # what a liveness-only floor does NOT calibrate the rate for
    not_certified: str           # what a ✅ riding a liveness-only floor does not certify
    confidence: str              # what the floor block gives confidence in
    fires: str                   # what a ✅ on a liveness-only floor does mean
    reliably: str                # what it does not


def liveness_line(label: str, fired: int, trials: int, wording: FloorWording, evidence: str) -> str:
    """The blockquote for a liveness-only floor. `evidence` completes "not evidence ..." and names
    what reliable capability would look like on this surface."""
    noun = wording.fire_nouns[0 if fired == 1 else 1]
    return (f"> ⚠️ liveness only (**{fired}/{trials}** < {GROUNDING_FLOOR_MIN_CORRECT}): the "
            f"**{label}** {wording.path} FIRES, but {fired} {noun} is a liveness proxy, not evidence "
            f"{evidence}. {wording.rate.format(label=label)} is calibrated for LIVENESS, not "
            f"{wording.calibrated_for} — raise this {wording.unit}'s capability trials/scenarios to "
            f">= {GROUNDING_FLOOR_MIN_CORRECT} to certify it.")


def liveness_footer(thin: set[str], wording: FloorWording) -> list[str]:
    """The note under the floor block when any floor is liveness-only (empty otherwise): the
    table's ✅ certifies non-degeneracy, not reliable capability."""
    if not thin:
        return []
    return [f"_⚠️ A liveness-only floor ({', '.join(sorted(thin))}) still yields a ✅ in the table "
            f"below — the table certifies NON-DEGENERACY (the {wording.unit} isn't dead), not "
            f"{wording.not_certified}. Read this floor block for {wording.confidence} confidence: a "
            f"✅ row riding a liveness-only floor means {wording.fires}, NOT that {wording.reliably}._",
            ""]
