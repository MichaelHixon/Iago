"""Central configuration — one place to change the target model and paths."""

from __future__ import annotations

from pathlib import Path

# Repo root is the parent of this package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTACKS_DIR = PROJECT_ROOT / "attacks"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "reports" / "artifacts"
OBJECTIVES_FILE = PROJECT_ROOT / "objectives.yaml"

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
    "refusal-suppression",
    "competing-objectives",
    "many-shot",
    "multi-turn",
    "rag-injection",
    "agent-abuse",
    "mcp-injection",
]
