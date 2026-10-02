"""Loud YAML readers for the declarative libraries (techniques, objectives, scenarios, frames).

`yaml.safe_load(...) or []` turned an empty or null file into zero records, so a truncated
library file shrank a run without a word, and a mapping where a list belonged was iterated by
its keys. Every loader reads through here instead, and a file that holds nothing usable raises
a ValueError naming it.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _describe(data: object) -> str:
    return "an empty or null document" if data is None else f"a {type(data).__name__}"


def load_records(path: Path) -> list[dict]:
    """A non-empty YAML list of mappings, or a ValueError naming `path`."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, list) or not data:
        got = "an empty list" if data == [] else _describe(data)
        raise ValueError(f"{path.name}: expected a non-empty YAML list of records, got {got}")
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"{path.name}: record {i} is a {type(rec).__name__}, not a mapping: "
                             f"{rec!r}")
    return data


def load_mapping(path: Path) -> dict:
    """A non-empty YAML mapping, or a ValueError naming `path`."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not data:
        got = "an empty mapping" if data == {} else _describe(data)
        raise ValueError(f"{path.name}: expected a non-empty YAML mapping, got {got}")
    return data
