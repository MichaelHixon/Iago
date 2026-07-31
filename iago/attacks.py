"""Attack library loader.

Techniques live as declarative YAML data under `attacks/` (one file per category),
so the library extends without touching Python. Each record wraps an `{objective}`
placeholder that the runner fills at execution time.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ATTACKS_DIR, CATEGORIES

REQUIRED_FIELDS = ("id", "name", "category", "description", "template")
PLACEHOLDER = "{objective}"

_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})


def _transform(objective: str, kind: str | None) -> str:
    """Actually encode the payload before injection (IMPROVEMENTS P2).

    Without this, an 'encoding' technique ships the objective in plain text and
    only tests instruction-following, not evasion — a real attacker sends the
    already-transformed payload past the surface filters.
    """
    if not kind or kind == "none":
        return objective
    if kind == "base64":
        return base64.b64encode(objective.encode()).decode()
    if kind == "reverse":
        return objective[::-1]
    if kind == "leetspeak":
        return objective.translate(_LEET)
    raise ValueError(f"unknown transform {kind!r}")


@dataclass(frozen=True)
class Technique:
    """One guardrail-bypass technique."""

    id: str
    name: str
    category: str
    description: str
    template: str
    reference: str | None = None
    transform: str | None = None  # none | base64 | reverse | leetspeak
    # OWASP LLM Top-10 tag — guardrail bypasses live under LLM01 (Prompt Injection),
    # which is the umbrella for jailbreaks. Override per technique in YAML when a
    # sharper mapping applies (e.g. LLM07 System-Prompt Leakage).
    owasp: str = "LLM01: Prompt Injection"
    # Multi-turn: priming user turns sent BEFORE the final `template` turn. The attack
    # builds across the conversation (crescendo / context-priming) so no single prompt
    # trips detection. None = single-shot. Priming turns may omit {objective}.
    turns: tuple[str, ...] | None = None

    @property
    def is_multiturn(self) -> bool:
        return bool(self.turns)

    def render(self, objective: str) -> str:
        """Encode the payload (if this technique transforms) then fill the placeholder."""
        payload = _transform(objective, self.transform)
        return self.template.replace(PLACEHOLDER, payload)

    def rendered_turns(self, objective: str) -> list[str]:
        """The full sequence of user turns: priming turns first, then the final ask."""
        payload = _transform(objective, self.transform)
        priming = [t.replace(PLACEHOLDER, payload) for t in (self.turns or ())]
        return priming + [self.template.replace(PLACEHOLDER, payload)]


def load_library(attacks_dir: Path | None = None) -> list[Technique]:
    """Read, validate, and return every technique across the attack YAML files.

    Fails loudly on a missing field, a template without the {objective} placeholder,
    an unknown category, or a duplicate id — a broken library should not run silently.
    """
    directory = Path(attacks_dir) if attacks_dir else ATTACKS_DIR
    techniques: list[Technique] = []
    seen_ids: set[str] = set()

    for path in sorted(directory.glob("*.yaml")):
        records = yaml.safe_load(path.read_text()) or []
        for rec in records:
            for field in REQUIRED_FIELDS:
                if field not in rec:
                    raise ValueError(f"{path.name}: technique missing '{field}': {rec!r}")
            if PLACEHOLDER not in rec["template"]:
                raise ValueError(
                    f"{path.name}: technique {rec['id']!r} template lacks the {PLACEHOLDER} placeholder"
                )
            if rec["category"] not in CATEGORIES:
                raise ValueError(
                    f"{path.name}: technique {rec['id']!r} has unknown category {rec['category']!r}"
                )
            if rec["id"] in seen_ids:
                raise ValueError(f"duplicate technique id {rec['id']!r} in {path.name}")
            seen_ids.add(rec["id"])
            techniques.append(
                Technique(
                    id=rec["id"],
                    name=rec["name"],
                    category=rec["category"],
                    description=rec["description"],
                    template=rec["template"],
                    reference=rec.get("reference"),
                    transform=rec.get("transform"),
                    owasp=rec.get("owasp", "LLM01: Prompt Injection"),
                    turns=tuple(rec["turns"]) if rec.get("turns") else None,
                )
            )

    return techniques


def summarize(techniques: list[Technique]) -> dict[str, int]:
    """Count techniques per category (for a quick library overview)."""
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for tech in techniques:
        counts[tech.category] += 1
    return counts


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    lib = load_library()
    print(f"Loaded {len(lib)} techniques across {len(CATEGORIES)} categories:")
    for category, count in summarize(lib).items():
        print(f"  {category:24} {count}")
