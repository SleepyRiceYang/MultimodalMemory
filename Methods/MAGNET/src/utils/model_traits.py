from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelTraits:
    """Encapsulate quirks of different action generators."""

    click_mode: str = "bbox"  # either "bbox" or "point"
    supports_stationary: bool = True


_MODEL_TRAITS = [
    ("gemini", ModelTraits(click_mode="bbox", supports_stationary=True)),
    ("qwenvl", ModelTraits(click_mode="bbox", supports_stationary=True)),
    ("atlas", ModelTraits(click_mode="point", supports_stationary=False)),
    ("guir1", ModelTraits(click_mode="point", supports_stationary=False)),
    ("ui_venus", ModelTraits(click_mode="point", supports_stationary=False)),
    ("infigui", ModelTraits(click_mode="point", supports_stationary=False)),
]


def get_model_traits(model_name: str, click_mode_override: str | None = None) -> ModelTraits:
    """Return the behavioural traits for a model, falling back to defaults."""
    if not model_name:
        return ModelTraits()
    lower = model_name.lower()
    for key, traits in _MODEL_TRAITS:
        if key in lower:
            base = traits
            break
    else:
        # Default assumption: GPT-4o/OpenAI-style models use bbox clicks and support stationary memory.
        base = ModelTraits()
    if click_mode_override:
        candidate = click_mode_override.lower()
        if candidate not in {"bbox", "point"}:
            raise ValueError(f"Unsupported click mode override: {click_mode_override}")
        base = replace(base, click_mode=candidate)
    # Default assumption: GPT-4o/OpenAI-style models use bbox clicks and support stationary memory.
    return base
