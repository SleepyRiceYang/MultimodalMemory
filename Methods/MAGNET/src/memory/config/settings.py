from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from memory.core.paths import repo_root

_CONFIG_ENV = "GUI_MEMORY_CONFIG"


@dataclass
class ModelDefaults:
    chunk_token_size: int
    chunk_overlap_token_size: int
    tiktoken_model_name: str
    llm_model_name: str
    llm_max_token_size: int
    llm_max_async: int
    embedding_batch_num: int
    embedding_func_max_async: int
    cosine_threshold: float


@dataclass
class LoggingConfig:
    rag_file: str
    gui_file: str


@dataclass
class PathSettings:
    cache_root: Path
    log_dir: Path
    data_root: Path
    storage_root: Path
    assets_root: Path


@dataclass
class ProjectSettings:
    paths: PathSettings
    rag_defaults: ModelDefaults
    gui_defaults: ModelDefaults
    workflow_defaults: ModelDefaults
    logging: LoggingConfig
    datasets: Dict[str, str]


def _resolve_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_model(data: Dict[str, Any]) -> ModelDefaults:
    return ModelDefaults(
        chunk_token_size=data["chunk_token_size"],
        chunk_overlap_token_size=data["chunk_overlap_token_size"],
        tiktoken_model_name=data["tiktoken_model_name"],
        llm_model_name=data["llm_model_name"],
        llm_max_token_size=data["llm_max_token_size"],
        llm_max_async=data["llm_max_async"],
        embedding_batch_num=data["embedding_batch_num"],
        embedding_func_max_async=data["embedding_func_max_async"],
        cosine_threshold=data["cosine_threshold"],
    )


def _parse_paths(data: Dict[str, Any]) -> PathSettings:
    return PathSettings(
        cache_root=_resolve_path(data["cache_root"]),
        log_dir=_resolve_path(data["log_dir"]),
        data_root=_resolve_path(data["data_root"]),
        storage_root=_resolve_path(data["storage_root"]),
        assets_root=_resolve_path(data["assets_root"]),
    )


def _parse_logging(data: Dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(**data)


def load_settings(path: str | Path | None = None) -> ProjectSettings:
    config_path = Path(path) if path else Path(os.environ.get(_CONFIG_ENV, repo_root() / "config" / "defaults.yaml"))
    payload = _load_yaml(config_path)

    paths_cfg = _parse_paths(payload.get("paths", {}))
    rag_defaults = _parse_model(payload.get("rag_defaults", {}))
    gui_defaults = _parse_model(payload.get("gui_defaults", {}))
    workflow_defaults = _parse_model(payload.get("workflow_defaults", {}))
    logging_cfg = _parse_logging(payload.get("logging", {}))
    datasets = payload.get("datasets", {})

    return ProjectSettings(
        paths=paths_cfg,
        rag_defaults=rag_defaults,
        gui_defaults=gui_defaults,
        workflow_defaults=workflow_defaults,
        logging=logging_cfg,
        datasets=datasets,
    )


_SETTINGS = load_settings()


def get_settings() -> ProjectSettings:
    return _SETTINGS
