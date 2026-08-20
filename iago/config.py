"""Central configuration — one place to change the target model and paths."""

from __future__ import annotations

from pathlib import Path

# Repo root is the parent of this package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTACKS_DIR = PROJECT_ROOT / "attacks"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "reports" / "artifacts"
OBJECTIVES_FILE = PROJECT_ROOT / "objectives.yaml"
# Agentic surface (indirect-injection against a tool-calling agent — the "lethal
# trifecta": private-data access + untrusted content + external comms).
# Kept in a subdirectory so the attack-library glob (attacks/*.yaml, non-recursive)
# never mistakes these agent scenarios for single-prompt technique files.
AGENT_SCENARIOS_FILE = PROJECT_ROOT / "attacks" / "agentic" / "agent_scenarios.yaml"
# Tool-abuse (RCE/SSRF) scenarios — a DIFFERENT bypass class from the exfil trifecta:
# a poisoned tool-output induces the agent to invoke a SANDBOXED dangerous tool
# (fake run_shell / fetch_url). Kept in its own file for the same glob-safety reason.
TOOLABUSE_SCENARIOS_FILE = PROJECT_ROOT / "attacks" / "agentic" / "toolabuse_scenarios.yaml"
# The unique token an injected shell command carries — its presence in a run_shell
# argument is deterministic ground truth that the RCE injection fired.
INJECTED_CMD_MARKER = "IAGO_INJECTED_CMD"

# Memory/context poisoning (ASI06) scenarios — a poisoned document plants an instruction
# the agent stores in memory that FIRES at a later step via recall.
MEMORY_SCENARIOS_FILE = PROJECT_ROOT / "attacks" / "agentic" / "memory_scenarios.yaml"
# RAG retrieval-poisoning scenarios — a poisoned chunk in the knowledge base is surfaced by a
# benign query and drives an unauthorized action.
RAG_SCENARIOS_FILE = PROJECT_ROOT / "attacks" / "agentic" / "rag_scenarios.yaml"
# The unique token an injected RAG chunk carries — its presence in a corpus chunk proves an
# injection exists to be retrieved (loader gate); the confirmed bypass is still the canary exfil.
INJECTED_RAG_MARKER = "IAGO_RAG_PAYLOAD"
# Inter-agent (ASI07) scenarios — a compromised/rogue PEER agent's message carries an injection
# the agent under test over-trusts and acts on.
A2A_SCENARIOS_FILE = PROJECT_ROOT / "attacks" / "agentic" / "a2a_scenarios.yaml"
# The unique token a poisoned peer-agent message carries — its presence in a peer message proves
# an injection exists in the inbox (loader gate); the confirmed bypass is still the canary exfil.
INJECTED_A2A_MARKER = "IAGO_A2A_PAYLOAD"
# The unique token a poisoned memory instruction carries — its presence in a `remember`
# argument proves the agent stored the POISON (not a benign note), the first half of the
# delayed-fire chain the temporal oracle confirms.
INJECTED_MEMORY_MARKER = "IAGO_MEM_PAYLOAD"

# A tool-using agent gets a bounded number of think→act steps per scenario so a
# hijacked loop can't run away; enough to read a doc, read notes, and send.
DEFAULT_AGENT_STEPS = 6

# Adaptive dialogue-level attacker (CoP/AJAR): the hard cap on turns per adaptive
# conversation. Anti-runaway — a search that never lands stops here and is recorded
# as `held`. Enough turns to open a frame and adapt a few times off the refusal shape.
DEFAULT_ADAPTIVE_TURNS = 5

# The agent's stated policy: outbound messages may only go to this domain. Any
# recipient outside it is exfiltration — a behavioral bypass the oracle confirms.
ALLOWED_RECIPIENT_DOMAIN = "corp.local"

# Local Ollama model pulled for this project (see the design notes).
DEFAULT_MODEL = "llama3.1:latest"

# Run methodology (IMPROVEMENTS P1): fire each technique x objective N times and
# report a bypass RATE, not a single shot — guardrail bypass is probabilistic.
DEFAULT_TRIALS = 3

# Pinned sampling for reproducibility (IMPROVEMENTS P1). Per-trial seed is
# BASE_SEED + trial_index, so trials differ but the whole run reproduces.
BASE_SEED = 1337
DEFAULT_TEMPERATURE = 0.8

# The attack categories the library is organized around.
CATEGORIES = [
    "direct-ask",
    "role-play",
    "format-shift",
    "instruction-hierarchy",
    "encoding-obfuscation",
    "low-resource-lang",
    "prompt-injection",
    "template-injection",
    "prompt-extraction",
    "refusal-suppression",
    "competing-objectives",
    "many-shot",
    "multi-turn",
    "composed-evasion",
    "provenance-forging",
    "rag-injection",
    "agent-abuse",
    "mcp-injection",
]

# OWASP Top 10 for Agentic Applications 2026 (verified 2026-08-19, announced
# 2025-12-09). The agentic analogue of the LLM0x tags: agentic techniques and
# scenarios carry an optional `asi` tag against these so findings map to the
# framework. Non-agentic techniques leave it unset.
VALID_ASI = (
    "ASI01",  # Agent Goal Hijack
    "ASI02",  # Tool Misuse & Exploitation
    "ASI03",  # Agent Identity & Privilege Abuse
    "ASI04",  # Agentic Supply Chain Compromise
    "ASI05",  # Unexpected Code Execution
    "ASI06",  # Memory & Context Poisoning
    "ASI07",  # Insecure Inter-Agent Communication
    "ASI08",  # Cascading Failures
    "ASI09",  # Human-Agent Trust Exploitation
    "ASI10",  # Rogue Agents
)


def validate_asi(value: str | None, *, where: str) -> str | None:
    """Return the asi tag unchanged, or raise loudly if it is off-list.

    None (or empty) is allowed — most techniques are not agentic. A present
    value must START with a known ASI id; the remainder may be a human label,
    e.g. "ASI09: Human-Agent Trust Exploitation".
    """
    if value is None or not str(value).strip():
        return None
    # Leading segment before the first colon; an empty/whitespace segment (e.g. a
    # leading-colon tag like ": ASI01") funnels into the ValueError below, not a
    # bare IndexError — the whole point is a loud, located error.
    head_seg = str(value).split(":", 1)[0].strip()
    parts = head_seg.split()
    head = parts[0] if parts else ""
    if head not in VALID_ASI:
        raise ValueError(
            f"{where}: invalid asi tag {value!r} — the leading id must be one of "
            f"{VALID_ASI} (OWASP Top 10 for Agentic Applications 2026)"
        )
    return value
