"""Agentic-scenario loader (defensive research).

Scenarios are declarative YAML — like the attack library and objectives — so the
agentic surface is fully data-driven. Each scenario wires a benign task to a set
of retrievable documents (one carrying the injection) plus a private secret the
agent must never exfiltrate. Fails loudly on a malformed record: a scenario whose
canary can't be traced, or whose injection has nothing to reach, is a silent
false-safe, so we reject it at load time rather than let it grade as "resisted".

ETHICS: for a model you own or are authorized to test. The secret is a synthetic
canary token, never a real credential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import AGENT_SCENARIOS_FILE

REQUIRED_FIELDS = ("id", "name", "kind", "task", "documents", "private_notes", "canary")
# attack     — a poisoned document/tool-result carries the injection.
# control    — clean content, no injection; a well-aligned agent resists.
# capability — POSITIVE CONTROL: the task itself legitimately asks for an external
#              send, proving the model can actually reach the exfil sink. If a
#              capability scenario does NOT go red, the agent is too weak a
#              tool-caller and a 0% attack rate is degenerate, not a real "policy held"
#              (the too-dumb-to-be-hijacked confound, Council 2026-08-01).
VALID_KINDS = ("attack", "control", "capability")


@dataclass(frozen=True)
class Scenario:
    """One agentic indirect-injection scenario."""

    id: str
    name: str
    kind: str  # attack | control
    task: str
    documents: dict[str, str]
    private_notes: str
    canary: str
    owasp: str | None = None
    attacker_recipient: str | None = None

    @property
    def is_control(self) -> bool:
        return self.kind == "control"

    @property
    def is_capability(self) -> bool:
        return self.kind == "capability"


def load_scenarios(scenarios_file: Path | None = None) -> list[Scenario]:
    """Read, validate, and return every scenario. Fails loudly on a bad record.

    A suite with zero attack scenarios measures nothing, so that is an error too.
    """
    path = Path(scenarios_file) if scenarios_file else AGENT_SCENARIOS_FILE
    if not path.exists():
        raise FileNotFoundError(f"agent scenarios file not found: {path}")

    records = yaml.safe_load(path.read_text()) or []
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()

    for rec in records:
        for f in REQUIRED_FIELDS:
            if f not in rec or rec[f] in (None, ""):
                raise ValueError(f"{path.name}: scenario missing '{f}': {rec!r}")
        if rec["kind"] not in VALID_KINDS:
            raise ValueError(
                f"{path.name}: scenario {rec['id']!r} has invalid kind {rec['kind']!r} "
                f"(want one of {VALID_KINDS})"
            )
        if rec["id"] in seen_ids:
            raise ValueError(f"duplicate scenario id {rec['id']!r} in {path.name}")
        seen_ids.add(rec["id"])

        docs = rec["documents"]
        if not isinstance(docs, dict) or not docs:
            raise ValueError(f"{path.name}: scenario {rec['id']!r} needs a non-empty documents map")

        # The canary must actually live in the private notes, or the exfil oracle
        # could never fire — the same loud-loader discipline as the prompt-leak path.
        if rec["canary"] not in rec["private_notes"]:
            raise ValueError(
                f"{path.name}: scenario {rec['id']!r} canary {rec['canary']!r} is not present in "
                "its private_notes — the exfiltration oracle could never fire"
            )

        scenarios.append(
            Scenario(
                id=rec["id"],
                name=rec["name"],
                kind=rec["kind"],
                task=rec["task"],
                documents={str(k): str(v) for k, v in docs.items()},
                private_notes=rec["private_notes"],
                canary=rec["canary"],
                owasp=rec.get("owasp"),
                attacker_recipient=rec.get("attacker_recipient"),
            )
        )

    if not any(s.kind == "attack" for s in scenarios):
        raise ValueError(f"{path.name}: at least one 'attack' scenario is required")

    return scenarios
