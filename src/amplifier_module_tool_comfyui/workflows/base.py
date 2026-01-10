"""Base workflow builder for ComfyUI."""

import random
from abc import ABC, abstractmethod
from typing import Any


class WorkflowBuilder(ABC):
    """Abstract base class for workflow builders."""

    @abstractmethod
    def build(self, **kwargs: Any) -> dict[str, Any]:
        """Build the workflow JSON."""
        ...

    @staticmethod
    def get_seed(seed: int | None) -> int:
        """Get seed value, generating random if not provided."""
        if seed is None:
            return random.randint(0, 2**32 - 1)
        return seed

    @staticmethod
    def create_node(class_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Create a workflow node."""
        return {
            "class_type": class_type,
            "inputs": inputs,
        }
