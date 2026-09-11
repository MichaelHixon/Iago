"""Iago — an authorized red-team harness for testing LLM guardrails.

Defensive research: attack library -> execute -> judge bypass -> pentest-style
report + hardening recommendations, against a local model you own. See the README.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("iago")  # single source of truth: pyproject.toml
except PackageNotFoundError:  # running from a bare checkout without an install
    __version__ = "0.0.0+unknown"
