from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from .schema import AgentConfig


def _read_yaml(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".json"}:
            return json.load(handle)
        return yaml.safe_load(handle)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Mapping[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_with_includes(config_path: Path, seen: set[Path] | None = None) -> Mapping[str, Any]:
    config_path = config_path.resolve()
    if seen is None:
        seen = set()
    if config_path in seen:
        raise ValueError(f"Cyclic config include detected at {config_path}")
    seen.add(config_path)

    data = _read_yaml(config_path) or {}
    includes = data.pop("INCLUDE", [])
    if isinstance(includes, str):
        includes = [includes]

    merged: Mapping[str, Any] = {}
    for inc in includes:
        inc_path = (config_path.parent / inc).resolve()
        included_data = _load_with_includes(inc_path, seen)
        merged = _deep_merge(merged, included_data)

    merged = _deep_merge(merged, data)
    seen.remove(config_path)
    return merged


def load_config(config_path: str | Path, overrides: MutableMapping[str, Any] | None = None) -> AgentConfig:
    config_path = Path(config_path)
    raw = _load_with_includes(config_path)
    cfg = AgentConfig.from_mapping(raw or {})
    cfg.merge_overrides(overrides)
    return cfg
