"""Pydantic models for ComfyUI tool requests and responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowType(str, Enum):
    """Supported workflow types."""

    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    UPSCALE = "upscale"


class GenerationRequest(BaseModel):
    """Request model for image generation."""

    prompt: str = Field(..., description="Text description of the image to generate")
    negative_prompt: str = Field(
        default="bad quality, blurry, distorted",
        description="What to avoid in the generated image",
    )
    width: int = Field(default=512, ge=64, le=2048, description="Image width in pixels")
    height: int = Field(
        default=512, ge=64, le=2048, description="Image height in pixels"
    )
    steps: int = Field(default=20, ge=1, le=150, description="Number of sampling steps")
    cfg_scale: float = Field(
        default=7.0, ge=1.0, le=30.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    sampler: str = Field(default="euler", description="Sampler algorithm to use")
    scheduler: str = Field(default="normal", description="Scheduler type")
    model: str | None = Field(
        default=None,
        description="Checkpoint model name (auto-detected if not specified)",
    )
    workflow: WorkflowType = Field(
        default=WorkflowType.TXT2IMG, description="Workflow type to use"
    )
    denoise: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Denoising strength (for img2img)"
    )
    input_image: str | None = Field(
        default=None, description="Base64 encoded input image (for img2img)"
    )
    output_format: str = Field(
        default="path", description="Output format: 'path' or 'base64'"
    )


class GenerationProgress(BaseModel):
    """Progress update during generation."""

    prompt_id: str
    node: str | None = None
    step: int | None = None
    total_steps: int | None = None
    preview: str | None = None  # Base64 preview if available


class GeneratedImage(BaseModel):
    """Result for a single generated image."""

    filename: str
    subfolder: str = ""
    type: str = "output"
    path: str | None = None  # Full path on server
    data: str | None = None  # Base64 encoded data


class GenerationResult(BaseModel):
    """Complete result of image generation."""

    prompt_id: str
    images: list[GeneratedImage]
    execution_time: float | None = None
    model_used: str | None = None


class ComfyUIStatus(BaseModel):
    """ComfyUI server status."""

    online: bool
    queue_remaining: int = 0
    queue_running: int = 0
    models_available: list[str] = Field(default_factory=list)
    system_stats: dict[str, Any] = Field(default_factory=dict)
