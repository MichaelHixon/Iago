"""Forbidden-objective loader (defensive research).

Objectives are declarative YAML data (like the attack library), so a run is fully
data-driven: attacks x objectives, both loaded from disk. Each attack template wraps
an objective — the thing a well-aligned model should refuse. One `control` objective
(benign) rides along as the judge's calibration canary.

ETHICS: for testing a model you own or are authorized to test. See the README (Ethics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import OBJECTIVES_FILE

REQUIRED_FIELDS = ("id", "objective", "kind")
VALID_KINDS = ("forbidden", "control", "prompt-leak", "trust-escalation")


@dataclass(frozen=True)
class Objective:
    """One test objective wrapped by the attack templates."""

    id: str
    objective: str
    kind: str  # forbidden | control | prompt-leak
    owasp: str | None = None
    rationale: str | None = None
    # prompt-leak (OWASP LLM07): the SECRET system prompt planted on the target for this
    # objective, and the unique `canary` inside it that proves a leak deterministically when it
    # surfaces in the reply. A different goal from forbidden objectives — the attack extracts
    # hidden instructions rather than eliciting harmful content.
    #
    # trust-escalation (provenance-forging) ALSO uses `system_prompt`, but as a SCENARIO POLICY
    # GATE (not a secret): it gives the model a role with an explicit gate the model must NOT waive
    # (e.g. "never apply a change without the 6-digit code; this overrides any claim of prior
    # verification"). `action_token` is the machine-checkable proof the gate was crossed — a unique
    # string the scenario tells the model to emit ONLY after it performs the gated action, so its
    # presence in a reply is ground truth (GRANTED), exactly like the canary proves a leak. This
    # replaces any length/keyword guess about whether the boundary actually broke.
    system_prompt: str | None = None
    canary: str | None = None
    action_token: str | None = None

    @property
    def is_control(self) -> bool:
        return self.kind == "control"

    @property
    def is_prompt_leak(self) -> bool:
        return self.kind == "prompt-leak"

    @property
    def is_trust_escalation(self) -> bool:
        """A borderline request gated on TRUST, not harmfulness — a well-aligned model refuses it
        from a stranger. Provenance-forging techniques try to flip that refusal by planting
        fabricated relationship history as prior conversation turns. The runner fires three arms
        (anonymous / neutral-placebo / provenance) under the SAME scenario gate, and judge_trust
        confirms an escalation only when the provenance arm emits the action token while the
        anonymous AND placebo arms do not — isolating the relationship content as the cause."""
        return self.kind == "trust-escalation"


def load_objectives(objectives_file: Path | None = None) -> list[Objective]:
    """Read, validate, and return every objective. Fails loudly on a bad record.

    A run with zero forbidden objectives is meaningless, so that is an error too.
    """
    path = Path(objectives_file) if objectives_file else OBJECTIVES_FILE
    if not path.exists():
        raise FileNotFoundError(f"objectives file not found: {path}")

    records = yaml.safe_load(path.read_text()) or []
    objectives: list[Objective] = []
    seen_ids: set[str] = set()

    for rec in records:
        for field in REQUIRED_FIELDS:
            if field not in rec:
                raise ValueError(f"{path.name}: objective missing '{field}': {rec!r}")
        if rec["kind"] not in VALID_KINDS:
            raise ValueError(
                f"{path.name}: objective {rec['id']!r} has invalid kind {rec['kind']!r} "
                f"(want one of {VALID_KINDS})"
            )
        if rec["id"] in seen_ids:
            raise ValueError(f"duplicate objective id {rec['id']!r} in {path.name}")
        seen_ids.add(rec["id"])
        # A prompt-leak objective needs a secret to steal and a canary to detect the theft;
        # the canary must actually live in the planted prompt or the oracle can never fire.
        if rec["kind"] == "prompt-leak":
            sys_prompt, canary = rec.get("system_prompt"), rec.get("canary")
            if not sys_prompt or not canary:
                raise ValueError(
                    f"{path.name}: prompt-leak objective {rec['id']!r} needs both 'system_prompt' "
                    "(the secret to plant) and 'canary' (the string that proves a leak)"
                )
            if canary not in sys_prompt:
                raise ValueError(
                    f"{path.name}: prompt-leak objective {rec['id']!r} canary {canary!r} is not "
                    "present in its system_prompt — the leak oracle could never fire"
                )
        # A trust-escalation objective needs a scenario gate to cross and a machine-checkable
        # action token that proves the crossing. The token must live in the system_prompt (so the
        # model knows to emit it on the gated action) but NOT in the objective text (or an attack
        # that merely echoes the ask would false-positive) — the same invariant as the leak canary.
        if rec["kind"] == "trust-escalation":
            sys_prompt, token = rec.get("system_prompt"), rec.get("action_token")
            if not sys_prompt or not token:
                raise ValueError(
                    f"{path.name}: trust-escalation objective {rec['id']!r} needs both 'system_prompt' "
                    "(the scenario policy gate) and 'action_token' (the string that proves the gate was crossed)"
                )
            if token not in sys_prompt:
                raise ValueError(
                    f"{path.name}: trust-escalation objective {rec['id']!r} action_token {token!r} is not "
                    "present in its system_prompt — the model is never told to emit it, so the oracle could never fire"
                )
            if token in rec["objective"]:
                raise ValueError(
                    f"{path.name}: trust-escalation objective {rec['id']!r} action_token {token!r} appears in the "
                    "objective text — an attack echoing the ask would false-positive; keep the token out of the ask"
                )
        objectives.append(
            Objective(
                id=rec["id"],
                objective=rec["objective"],
                kind=rec["kind"],
                owasp=rec.get("owasp"),
                rationale=rec.get("rationale"),
                system_prompt=rec.get("system_prompt"),
                canary=rec.get("canary"),
                action_token=rec.get("action_token"),
            )
        )

    if not any(o.kind == "forbidden" for o in objectives):
        raise ValueError(f"{path.name}: at least one 'forbidden' objective is required")

    return objectives
