from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping

from .base import BaseModel
from .gemini import GeminiModel
from .openai import OpenAIModel
from .qwen import QwenVLModel


MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "gemini": GeminiModel,
    "qwenvl": QwenVLModel,
    "atlas": QwenVLModel,
    "ui-venus": QwenVLModel,
    "infigui-r1": QwenVLModel,
}


def _ensure_mapping(config: Any) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    raise TypeError(f"Unsupported model config type: {type(config)}")


def get_model(model_config: Dict[str, Any], seed: int = 2024) -> BaseModel:
    cfg = _ensure_mapping(model_config)
    model_name = cfg["NAME"].lower()
    for key, cls in MODEL_REGISTRY.items():
        if key in model_name:
            return cls(cfg, seed=seed)
    raise NotImplementedError(f"Unsupported model {cfg['NAME']}")
