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
VALID_KINDS = ("forbidden", "control")


@dataclass(frozen=True)
class Objective:
    """One test objective wrapped by the attack templates."""

    id: str
    objective: str
    kind: str  # forbidden | control
    owasp: str | None = None
    rationale: str | None = None

    @property
    def is_control(self) -> bool:
        return self.kind == "control"


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
        objectives.append(
            Objective(
                id=rec["id"],
                objective=rec["objective"],
                kind=rec["kind"],
                owasp=rec.get("owasp"),
                rationale=rec.get("rationale"),
            )
        )

    if not any(o.kind == "forbidden" for o in objectives):
        raise ValueError(f"{path.name}: at least one 'forbidden' objective is required")

    return objectives
