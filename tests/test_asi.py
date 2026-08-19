"""OWASP Agentic (ASI) Top-10 tagging.

The agentic surfaces (agent-abuse / mcp-injection / rag-injection techniques and
the indirect-injection scenarios) carry an optional `asi` tag against the OWASP
Top 10 for Agentic Applications 2026, the way single-turn techniques carry
`owasp`. These tests lock the loud validation, the load-through, and the fact
that non-agentic techniques stay untagged.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from iago.attacks import load_library
from iago.agent_scenarios import load_scenarios
from iago.config import VALID_ASI, validate_asi


def test_validate_asi_accepts_bare_and_labelled():
    assert validate_asi("ASI09", where="t") == "ASI09"
    assert validate_asi("ASI09: Human-Agent Trust Exploitation", where="t") == (
        "ASI09: Human-Agent Trust Exploitation"
    )


def test_validate_asi_allows_absent():
    assert validate_asi(None, where="t") is None
    assert validate_asi("  ", where="t") is None


def test_validate_asi_rejects_offlist_loudly():
    with pytest.raises(ValueError, match="invalid asi tag"):
        validate_asi("ASI11: Not A Thing", where="scenario x")
    with pytest.raises(ValueError, match="invalid asi tag"):
        validate_asi("LLM01: Prompt Injection", where="scenario x")
    # Leading-colon / empty-head input must funnel into the loud ValueError,
    # never a bare IndexError (Council + code-review catch).
    for bad in (":", ": ASI01", "  :  x", "::"):
        with pytest.raises(ValueError, match="invalid asi tag"):
            validate_asi(bad, where="scenario x")


def test_all_ten_ids_present():
    assert VALID_ASI == tuple(f"ASI{n:02d}" for n in range(1, 11))


def test_agentic_techniques_carry_asi_nonagentic_do_not():
    lib = {t.id: t for t in load_library()}
    # Agentic techniques are tagged with the mapped ASI id.
    assert lib["agent-001"].asi.startswith("ASI02")  # poisoned tool output → tool misuse
    assert lib["agent-002"].asi.startswith("ASI01")  # goal substitution → goal hijack
    assert lib["mcp-001"].asi.startswith("ASI04")    # poisoned manifest → supply chain
    assert lib["mcp-002"].asi.startswith("ASI02")    # server response → tool misuse
    assert lib["rag-001"].asi.startswith("ASI06")    # retrieved-doc → memory/context poisoning
    assert lib["rag-002"].asi.startswith("ASI06")
    # Single-turn jailbreaks are model-level, not agentic — they carry no ASI tag.
    non_agentic = [t for t in load_library()
                   if t.category not in ("agent-abuse", "mcp-injection", "rag-injection")]
    assert non_agentic, "expected some non-agentic techniques in the library"
    assert all(t.asi is None for t in non_agentic)


def test_scenarios_carry_asi():
    scens = {s.id: s for s in load_scenarios()}
    assert scens["exfil-via-doc"].asi.startswith("ASI01")
    assert scens["shadow-recipient"].asi.startswith("ASI01")  # recipient override → goal hijack
    # Controls / capability floors make no agentic-risk claim — left untagged.
    assert scens["benign-summary"].asi is None
    assert scens["capability-floor"].asi is None


def test_technique_loader_rejects_bad_asi(tmp_path: Path):
    bad = tmp_path / "bad_asi.yaml"
    bad.write_text(
        textwrap.dedent(
            """\
            - id: bad-001
              name: Bad ASI tag
              category: agent-abuse
              description: x
              asi: "ASI99: nope"
              template: "do {objective}"
            """
        )
    )
    with pytest.raises(ValueError, match="invalid asi tag"):
        load_library(bad.parent)
