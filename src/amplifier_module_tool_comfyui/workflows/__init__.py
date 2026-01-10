"""Workflow templates for ComfyUI."""

from .base import WorkflowBuilder
from .img2img import Img2ImgWorkflow
from .ip_adapter import IPAdapterWorkflow
from .sdxl import SDXLWorkflow
from .txt2img import Txt2ImgWorkflow
from .txt2img_lora import Txt2ImgLoRAWorkflow
from .upscale import UpscaleWorkflow

__all__ = [
    "WorkflowBuilder",
    "Txt2ImgWorkflow",
    "Txt2ImgLoRAWorkflow",
    "Img2ImgWorkflow",
    "UpscaleWorkflow",
    "SDXLWorkflow",
    "IPAdapterWorkflow",
]
