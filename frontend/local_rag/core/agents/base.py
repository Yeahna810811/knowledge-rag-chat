from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Structured output from a single agent step."""

    agent: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class BaseAgent(ABC):
    """Base class for collaborative RAG agents."""

    name: str = "base_agent"

    @abstractmethod
    def run(self, **kwargs: Any) -> AgentResult:
        raise NotImplementedError
