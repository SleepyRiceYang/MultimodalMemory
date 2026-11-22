from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Type, TypeVar, get_type_hints


class _AttrDictMixin:
    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass(slots=True)
class DataSettings(_AttrDictMixin):
    DATASET: str = ""
    DATA_DIR: str = ""
    SPLIT: str = "test"
    USE_SCREEN_TAG: bool = True
    BBOX_FORMAT: str = "relative_int"


@dataclass(slots=True)
class ModelSettings(_AttrDictMixin):
    NAME: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_API_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o-2024-11-20"
    CLICK_MODE: str | None = None
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-pro-vision"
    QWENVL_API_KEY: str = ""
    QWENVL_API_URL: str = ""
    QWENVL_MODEL: str = "qwen2.5-vl-72b-instruct"
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_MODEL: str = "qwen-vl-max"
    MAX_IMAGE_SIDE: int = 1024


@dataclass(slots=True)
class MemorySettings(_AttrDictMixin):
    STATIONARY: bool = False
    STATIONARY_THRESHOLD: float = 0.7
    PROCEDURAL: bool = False
    PROCEDURAL_THRESHOLD: float = 0.4
    STATIONARY_RAG_DIRS: Dict[str, str] = field(default_factory=dict)
    PROCEDURAL_RAG_DIR: str | None = None
    INHERIT_COUNTER: bool = True
    HINT_ENABLE: bool = False
    HINT_K: int = 5
    TM_THRESHOLD: float = 0.3
    OMNIPARSER_CONFIG: str = ""


T = TypeVar("T")


def _default_for(f):
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


def _dataclass_from_mapping(cls: Type[T], payload: Mapping[str, Any] | None) -> T:
    payload = payload or {}
    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        field_type = type_hints.get(f.name, f.type)
        value = payload.get(f.name)
        if is_dataclass(field_type):
            kwargs[f.name] = _dataclass_from_mapping(field_type, value or {})
        else:
            kwargs[f.name] = value if value is not None else _default_for(f)
    return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class AgentConfig(_AttrDictMixin):
    DEMO_NAME: str = "CoATAgent"
    DEMO_MODE: str = "COAT"
    OUTPUT_DIR: str = "output"
    METRICS_DIR: str = "metrics"
    LOG_PROMPTS: bool = False
    DATA: DataSettings = field(default_factory=DataSettings)
    MODEL: ModelSettings = field(default_factory=ModelSettings)
    MEMORY: MemorySettings = field(default_factory=MemorySettings)
    PROMPTS: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AgentConfig":
        kwargs: Dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for f in fields(cls):
            field_type = type_hints.get(f.name, f.type)
            value = payload.get(f.name)
            if is_dataclass(field_type):
                kwargs[f.name] = _dataclass_from_mapping(field_type, value)
            else:
                kwargs[f.name] = value if value is not None else _default_for(f)
        return cls(**kwargs)  # type: ignore[arg-type]

    def merge_overrides(self, overrides: MutableMapping[str, Any] | None):
        if not overrides:
            return
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if is_dataclass(current) and isinstance(value, Mapping):
                for sub_key, sub_val in value.items():
                    if hasattr(current, sub_key):
                        setattr(current, sub_key, sub_val)
            else:
                setattr(self, key, value)
