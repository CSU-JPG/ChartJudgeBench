from __future__ import annotations

from abc import ABC, abstractmethod

from ..protocols import EvaluationRequest


class ModelAdapter(ABC):
    """Minimal interface required to evaluate any multimodal model."""

    thread_safe = False

    @abstractmethod
    def generate(self, request: EvaluationRequest) -> str:
        """Return the model's raw text response for one multimodal request."""

    def close(self) -> None:
        """Release optional backend resources."""
