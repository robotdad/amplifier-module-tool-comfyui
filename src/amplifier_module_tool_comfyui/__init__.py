"""Amplifier ComfyUI Tool Module.

This module provides image generation capabilities via ComfyUI's REST/WebSocket API.

Usage in mount plan:
    tools:
      - module: comfyui
        source: git+https://github.com/robotdad/amplifier-module-tool-comfyui@main
        config:
          base_url: "http://127.0.0.1:8188"
          timeout: 300
          default_model: "v1-5-pruned-emaonly.safetensors"
"""

from typing import Any, Callable

from .client import ComfyUIClient
from .models import (
    ComfyUIStatus,
    GeneratedImage,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    WorkflowType,
)
from .tool import ComfyUITool

__all__ = [
    "ComfyUIClient",
    "ComfyUITool",
    "ComfyUIStatus",
    "GeneratedImage",
    "GenerationProgress",
    "GenerationRequest",
    "GenerationResult",
    "WorkflowType",
    "mount",
]

__version__ = "0.1.0"


async def mount(coordinator: Any, config: dict[str, Any]) -> Callable[[], Any]:
    """Mount the ComfyUI tool into the Amplifier coordinator.

    Args:
        coordinator: Amplifier ModuleCoordinator instance
        config: Tool configuration dictionary with keys:
            - base_url: ComfyUI server URL (default: http://127.0.0.1:8188)
            - timeout: Request timeout in seconds (default: 300)
            - default_model: Default checkpoint model name
            - output_format: Default output format ('path' or 'base64')
            - use_websocket: Use WebSocket for progress (default: True)

    Returns:
        Cleanup function to call when unmounting
    """
    tool = ComfyUITool(config=config)
    await coordinator.mount("tools", tool, name=tool.name)

    # Return cleanup function
    return tool.cleanup
