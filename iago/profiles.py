"""Declarative, version-stamped model profiles — the model-skin half of a `(technique, profile)` pair.

A profile names a target model FAMILY and the per-family knobs a technique's rendering layer honors:
the control-channel format the family respects, the refusal-vocabulary markers its guardrail emits,
and which carriers (transforms + wrapper frames) are effective against it. A profile carries NO
operative payload body (ISC-65): no template, objective, or response text ever lives here — only
technique parameters and markers, exactly as `judge_control_set.jsonl` keeps forbidden bodies as
hashes. Loading a profile with no `model_version` stamp raises rather than running against stale
tuning (ISC-62).

Modeled on `attacks.load_library`: a frozen dataclass, a loud loader validating every record, and
the same carrier registries so a profile can only name carriers the render pipeline actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import sha256_text
from .attacks import TRANSFORM_KINDS, wrapper_frame_names
from .config import PROFILES_DIR
from .yaml_records import load_records

# Every field a profile MUST declare. `model_version` is load-bearing: an unstamped profile is
# stale-tuning-prone and raises (ISC-62).
REQUIRED_PROFILE_FIELDS = ("id", "name", "description", "family", "model_version", "control_channel")

# The control-channel formats a family can honor. Closed vocabulary so a typo fails loudly at load
# instead of silently rendering the family-blind default.
PROFILE_CONTROL_CHANNELS: frozenset[str] = frozenset(
    {"none", "system-role", "developer-role", "inst-tags", "chatml", "header-prefix"}
)

# Keys that would smuggle an operative payload body into a profile. The loader refuses any of them
# (ISC-65) — a profile is parameters and markers, never a working prompt.
FORBIDDEN_BODY_FIELDS: frozenset[str] = frozenset(
    {"template", "objective", "response", "prompt", "payload", "body", "text", "turns", "shots"}
)

# The complete set of keys a profile record may carry. An unrecognized key — above all a near-miss
# typo like `refusal_marker`/`carrier` — must RAISE, not load a silently-empty profile: in a
# loud-loader codebase a silent drop is the worst failure mode. Optional keys are the two list
# fields; everything else is required.
ALLOWED_PROFILE_FIELDS: frozenset[str] = frozenset(REQUIRED_PROFILE_FIELDS) | {
    "refusal_markers", "carriers",
}

# ISC-65 is a VALUE invariant, not only a key invariant: a free-text field can carry an operative
# payload under a legitimate key. The free-text channels (`description`, each `refusal_markers`
# entry) are length-capped so they are too small to ship a paste-ready jailbreak scaffold. A marker
# is a short refusal phrase; a description is a one- or two-sentence summary. (A multi-entry
# evasion spread across many short markers is still visibly a marker list and is caught at push
# time by the public-scrub gate + human review — the caps close the single-field body channel.)
MAX_MARKER_LEN = 80
MAX_DESCRIPTION_LEN = 400


@dataclass(frozen=True)
class ModelProfile:
    """One model skin: target family + the render knobs a technique honors for it."""

    id: str
    name: str
    description: str
    family: str
    model_version: str
    control_channel: str
    refusal_markers: tuple[str, ...] = ()
    carriers: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        """A stable id for the profile's DECLARED config, stamped onto every tuned row as
        `model_profile_sha256` so a reader can pin a quoted render to the exact skin that made it.

        It is a declared-config stamp, not a model-build guard: it changes when any operative field
        here changes (version, channel, markers, carriers), but it cannot detect an upstream model
        swapped out under the SAME `model_version` string — nothing here observes the model's weights.
        """
        # \x1f delimits both the outer parts AND the inner list joins, so the hash is injective:
        # ["a", "b"] and ["a|b"] must not collide (an in-band "|" would let them).
        parts = (self.id, self.family, self.model_version, self.control_channel,
                 "\x1f".join(self.refusal_markers), "\x1f".join(self.carriers))
        return sha256_text("\x1f".join(parts))


def _valid_carriers() -> frozenset[str]:
    """Carriers a profile may name: the transform kinds plus the wrapper frame names the render
    pipeline actually understands. A profile naming anything else fails loudly at load."""
    return TRANSFORM_KINDS | wrapper_frame_names()


def load_profiles(profiles_dir: Path | None = None) -> list[ModelProfile]:
    """Read, validate, and return every profile across the profile YAML files.

    Fails loudly on a missing required field (including `model_version` — ISC-62), an unknown
    control channel, an unknown carrier, a duplicate id, or any payload-body-shaped key (ISC-65).
    """
    directory = Path(profiles_dir) if profiles_dir else PROFILES_DIR
    carriers_ok = _valid_carriers()
    profiles: list[ModelProfile] = []
    seen_ids: set[str] = set()

    for path in sorted(directory.glob("*.yaml")):
        for rec in load_records(path):
            for field in REQUIRED_PROFILE_FIELDS:
                if field not in rec:
                    raise ValueError(f"{path.name}: profile missing '{field}': {rec!r}")
                if not isinstance(rec[field], str) or not rec[field]:
                    # A non-string stamp (e.g. an unquoted `model_version: 2` parsed as int) must
                    # fail at load, not later in `fingerprint` — ISC-62's stale-tuning guard only
                    # holds if the version is a real string.
                    raise ValueError(
                        f"{path.name}: profile {rec.get('id')!r} field {field!r} must be a "
                        f"non-empty string, got {rec[field]!r}"
                    )
            forbidden = FORBIDDEN_BODY_FIELDS & set(rec)
            if forbidden:
                raise ValueError(
                    f"{path.name}: profile {rec['id']!r} carries payload-body field(s) "
                    f"{sorted(forbidden)} — profiles hold parameters and markers only (ISC-65)"
                )
            unknown = set(rec) - ALLOWED_PROFILE_FIELDS
            if unknown:
                # A near-miss typo (`refusal_marker`, `carrier`) must RAISE, not load a
                # silently-empty profile — a silent drop is the worst failure in a loud loader.
                raise ValueError(
                    f"{path.name}: profile {rec['id']!r} has unknown field(s) {sorted(unknown)} "
                    f"(allowed: {sorted(ALLOWED_PROFILE_FIELDS)})"
                )
            if len(rec["description"]) > MAX_DESCRIPTION_LEN:
                raise ValueError(
                    f"{path.name}: profile {rec['id']!r} description is {len(rec['description'])} "
                    f"chars (max {MAX_DESCRIPTION_LEN}) — a profile is a short skin, not a body (ISC-65)"
                )
            if rec["control_channel"] not in PROFILE_CONTROL_CHANNELS:
                raise ValueError(
                    f"{path.name}: profile {rec['id']!r} has unknown control_channel "
                    f"{rec['control_channel']!r} (want one of {sorted(PROFILE_CONTROL_CHANNELS)})"
                )
            for list_field in ("refusal_markers", "carriers"):
                value = rec.get(list_field)
                if value is not None and not isinstance(value, list):
                    raise ValueError(
                        f"{path.name}: profile {rec['id']!r} field {list_field!r} must be a list, "
                        f"got {type(value).__name__}"
                    )
            markers = tuple(rec.get("refusal_markers") or ())
            for marker in markers:
                if not isinstance(marker, str) or len(marker) > MAX_MARKER_LEN:
                    raise ValueError(
                        f"{path.name}: profile {rec['id']!r} refusal_marker {marker!r} is not a "
                        f"string of <= {MAX_MARKER_LEN} chars — markers are short phrases (ISC-65)"
                    )
            carriers = tuple(rec.get("carriers") or ())
            for carrier in carriers:
                if carrier not in carriers_ok:
                    raise ValueError(
                        f"{path.name}: profile {rec['id']!r} names unknown carrier {carrier!r} "
                        f"(want a subset of {sorted(carriers_ok)})"
                    )
            if rec["id"] in seen_ids:
                raise ValueError(f"duplicate profile id {rec['id']!r} in {path.name}")
            seen_ids.add(rec["id"])
            profiles.append(ModelProfile(
                id=rec["id"], name=rec["name"], description=rec["description"],
                family=rec["family"], model_version=rec["model_version"],
                control_channel=rec["control_channel"],
                refusal_markers=markers, carriers=carriers,
            ))

    return profiles
