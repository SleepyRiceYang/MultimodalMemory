from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from memory.core.paths import storage_path


STATIONARY_RAG_DEFAULTS = {
    "odyssey": "odyssey_stationary_omni",
    "android-in-the-zoo": "aitz_stationary_omni",
    "amex": "amex_stationary_omni",
    "screenspot": "storage_omni",
}

PROCEDURAL_RAG_DEFAULTS = {
    "odyssey": "odyssey_procedural",
    "android-in-the-zoo": "aitz_procedural",
    "amex": "amex_procedural",
}


def _get_value(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, "__getitem__"):
        try:
            return container[key]
        except Exception:
            pass
    return getattr(container, key, default)


def _resolve_rag_path(candidate: Optional[str], default: Optional[str]) -> Optional[str]:
    if candidate:
        return str(Path(candidate).expanduser())
    if default:
        return str(storage_path(default))
    return None


def build_stationary_memory(config: Dict[str, Any], embedding_config):
    memory_cfg = _get_value(config, "MEMORY", {})
    data_cfg = _get_value(config, "DATA", {})

    if not _get_value(memory_cfg, "STATIONARY", False):
        return None

    dataset = _get_value(data_cfg, "DATASET", "")
    rag_overrides = _get_value(memory_cfg, "STATIONARY_RAG_DIRS", {}) or {}

    rag_dir = _resolve_rag_path(
        rag_overrides.get("default"),
        STATIONARY_RAG_DEFAULTS.get(dataset),
    )

    from memory.stores.stationary_mem import StationaryMemory

    kwargs = {"threshold": _get_value(memory_cfg, "STATIONARY_THRESHOLD", 0.7)}

    if rag_dir is None:
        raise ValueError(f"No stationary memory directory configured for dataset '{dataset}'")

    print("Loading stationary memory from:", rag_dir)
    inherit_counter = _get_value(memory_cfg, "INHERIT_COUNTER", True)
    return StationaryMemory(
        "hf",
        "MiniCPM-V-2_6",
        embedding_config,
        model_dir="openbmb",
        rag_dir=rag_dir,
        inherit_counter=inherit_counter,
        **kwargs,
    )


def build_procedural_memory(config: Dict[str, Any], embedding_config):
    memory_cfg = _get_value(config, "MEMORY", {})
    data_cfg = _get_value(config, "DATA", {})

    if not _get_value(memory_cfg, "PROCEDURAL", False):
        return None

    from memory.stores.procedural_mem import ProceduralMemory

    dataset = _get_value(data_cfg, "DATASET", "")
    rag_dir = _resolve_rag_path(
        _get_value(memory_cfg, "PROCEDURAL_RAG_DIR"),
        PROCEDURAL_RAG_DEFAULTS.get(dataset),
    )
    if rag_dir is None:
        raise ValueError(f"No procedural memory directory configured for dataset '{dataset}'")

    print("Loading procedural memory from:", rag_dir)
    inherit_counter = _get_value(memory_cfg, "INHERIT_COUNTER", True)
    return ProceduralMemory(
        "hf",
        "MiniCPM-V-2_6",
        embedding_config,
        threshold=_get_value(memory_cfg, "PROCEDURAL_THRESHOLD", 0.4),
        model_dir="openbmb",
        rag_dir=rag_dir,
        inherit_counter=inherit_counter,
    )
