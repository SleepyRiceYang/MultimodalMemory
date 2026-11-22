from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract interface all vision-language model connectors must follow."""

    @abstractmethod
    def format_query(self, *args: Any, **kwargs: Any):
        """Convert the agent prompts into provider specific payload."""

    @abstractmethod
    def get_response(self, *args: Any, **kwargs: Any):
        """Single shot request to the underlying model."""

    @abstractmethod
    def chat(self, *args: Any, **kwargs: Any):
        """Interactive chat style interface."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_response(*args, **kwargs)
