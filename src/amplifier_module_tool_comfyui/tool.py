"""ComfyUI Tool implementation for Amplifier."""

import base64
from typing import Any

from amplifier_core import ToolResult

from .client import ComfyUIClient
from .models import GenerationRequest, WorkflowType
from .workflows import (
    Img2ImgWorkflow,
    IPAdapterWorkflow,
    SDXLWorkflow,
    Txt2ImgLoRAWorkflow,
    Txt2ImgWorkflow,
    UpscaleWorkflow,
)


class ComfyUITool:
    """Amplifier tool for generating images via ComfyUI.

    This tool provides LLM agents with image generation capabilities using
    Stable Diffusion models through ComfyUI's REST/WebSocket API.

    Supports:
    - Text-to-image generation
    - Image-to-image transformation
    - Image upscaling

    Configuration:
        base_url: ComfyUI server URL (default: http://127.0.0.1:8188)
        timeout: Request timeout in seconds (default: 300)
        default_model: Default checkpoint model name
        output_format: Default output format ('path' or 'base64')
        use_websocket: Use WebSocket for progress (default: True)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.base_url = config.get("base_url", "http://127.0.0.1:8188")
        self.timeout = config.get("timeout", 300.0)
        self.default_model = config.get("default_model", None)
        self.output_format = config.get("output_format", "path")
        self.use_websocket = config.get("use_websocket", True)

        self._client = ComfyUIClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

        # Workflow builders
        self._txt2img = Txt2ImgWorkflow()
        self._txt2img_lora = Txt2ImgLoRAWorkflow()
        self._img2img = Img2ImgWorkflow()
        self._upscale = UpscaleWorkflow()
        self._sdxl = SDXLWorkflow()
        self._ip_adapter = IPAdapterWorkflow()

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return """Generate images using Stable Diffusion via ComfyUI.

Workflow types:
- txt2img: Generate images from text descriptions
- img2img: Transform existing images based on prompts  
- upscale: Upscale images with optional refinement
- sdxl: High-quality generation at 1024x1024 using SDXL models
- ip_adapter: Use reference images to guide style/subject

Basic parameters:
- prompt (required): Text description of the desired image
- negative_prompt: What to avoid (default: "bad quality, blurry, distorted")
- width/height: Image dimensions (default: 512, or 1024 for SDXL)
- steps: Sampling steps (default: 20)
- cfg_scale: Guidance scale (default: 7.0)
- seed: Random seed for reproducibility
- sampler: Sampling algorithm (euler, dpmpp_2m, ddim, etc.)
- model: Checkpoint model name

LoRA parameters (works with any workflow):
- lora_name: LoRA model filename to apply
- lora_strength: LoRA influence (0-2, default: 1.0)

SDXL parameters (workflow: "sdxl"):
- refiner_model: Optional SDXL refiner for quality boost
- refiner_start: When to switch to refiner (0-1, default: 0.8)

IP-Adapter parameters (workflow: "ip_adapter"):
- reference_image: Base64 encoded reference image for style/subject
- ip_adapter_weight: Reference influence strength (0-2, default: 1.0)

Examples:
{"prompt": "a majestic mountain at sunset", "steps": 25}
{"prompt": "portrait in anime style", "lora_name": "anime_style.safetensors", "lora_strength": 0.8}
{"workflow": "sdxl", "prompt": "photorealistic landscape", "width": 1024, "height": 1024}"""

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid in the generated image",
                    "default": "bad quality, blurry, distorted",
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels",
                    "default": 512,
                    "minimum": 64,
                    "maximum": 2048,
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels",
                    "default": 512,
                    "minimum": 64,
                    "maximum": 2048,
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of sampling steps",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 150,
                },
                "cfg_scale": {
                    "type": "number",
                    "description": "Classifier-free guidance scale",
                    "default": 7.0,
                    "minimum": 1.0,
                    "maximum": 30.0,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                },
                "workflow": {
                    "type": "string",
                    "description": "Workflow type",
                    "enum": ["txt2img", "img2img", "upscale", "sdxl", "ip_adapter"],
                    "default": "txt2img",
                },
                "input_image": {
                    "type": "string",
                    "description": "Base64 encoded input image (for img2img/upscale)",
                },
                "denoise": {
                    "type": "number",
                    "description": "Denoising strength (0-1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "model": {
                    "type": "string",
                    "description": "Checkpoint model name",
                },
                "sampler": {
                    "type": "string",
                    "description": "Sampling algorithm",
                    "enum": [
                        "euler",
                        "euler_ancestral",
                        "heun",
                        "dpm_2",
                        "dpm_2_ancestral",
                        "lms",
                        "dpm_fast",
                        "dpm_adaptive",
                        "dpmpp_2s_ancestral",
                        "dpmpp_sde",
                        "dpmpp_2m",
                        "dpmpp_2m_sde",
                        "ddim",
                        "uni_pc",
                    ],
                    "default": "euler",
                },
                # LoRA parameters
                "lora_name": {
                    "type": "string",
                    "description": "LoRA model filename to apply for style/character",
                },
                "lora_strength": {
                    "type": "number",
                    "description": "LoRA model influence strength",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
                "lora_clip_strength": {
                    "type": "number",
                    "description": "LoRA CLIP influence strength",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
                # SDXL parameters
                "refiner_model": {
                    "type": "string",
                    "description": "SDXL refiner model for quality enhancement",
                },
                "refiner_start": {
                    "type": "number",
                    "description": "When to switch to refiner (0-1)",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                # IP-Adapter parameters
                "reference_image": {
                    "type": "string",
                    "description": "Base64 encoded reference image for style/subject transfer",
                },
                "ip_adapter_model": {
                    "type": "string",
                    "description": "IP-Adapter model name",
                    "default": "ip-adapter_sd15.safetensors",
                },
                "ip_adapter_weight": {
                    "type": "number",
                    "description": "IP-Adapter influence strength",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute image generation."""
        try:
            # Parse and validate input
            request = self._parse_input(input)

            # Check server availability
            if not await self._client.health_check():
                return ToolResult(
                    success=False,
                    error={
                        "message": f"ComfyUI server not available at {self.base_url}",
                        "type": "ConnectionError",
                    },
                )

            # Get model to use
            model = await self._resolve_model(request.model)
            if not model:
                return ToolResult(
                    success=False,
                    error={
                        "message": "No models available. Please install a checkpoint model.",
                        "type": "ConfigurationError",
                    },
                )

            # Handle input image for img2img/upscale
            input_image_name = None
            if request.workflow in (WorkflowType.IMG2IMG, WorkflowType.UPSCALE):
                if not request.input_image:
                    return ToolResult(
                        success=False,
                        error={
                            "message": f"input_image is required for {request.workflow.value} workflow",
                            "type": "ValidationError",
                        },
                    )
                input_image_name = await self._upload_input_image(request.input_image)

            # Handle reference image for IP-Adapter
            reference_image_name = None
            if request.workflow == WorkflowType.IP_ADAPTER:
                if not request.reference_image:
                    return ToolResult(
                        success=False,
                        error={
                            "message": "reference_image is required for ip_adapter workflow",
                            "type": "ValidationError",
                        },
                    )
                reference_image_name = await self._upload_input_image(
                    request.reference_image
                )

            # Build workflow
            workflow = self._build_workflow(
                request, model, input_image_name, reference_image_name
            )

            # Execute generation
            result = await self._client.generate_and_fetch(
                workflow=workflow,
                output_format=request.output_format,
                use_websocket=self.use_websocket,
            )

            # Format response
            images_output = []
            for img in result.images:
                img_info = {
                    "filename": img.filename,
                    "subfolder": img.subfolder,
                    "type": img.type,
                }
                if img.data:
                    img_info["data"] = img.data
                images_output.append(img_info)

            return ToolResult(
                success=True,
                output={
                    "prompt_id": result.prompt_id,
                    "images": images_output,
                    "model_used": model,
                    "workflow": request.workflow.value,
                    "message": f"Generated {len(images_output)} image(s) using {request.workflow.value}",
                },
            )

        except TimeoutError as e:
            return ToolResult(
                success=False,
                error={
                    "message": str(e),
                    "type": "TimeoutError",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error={
                    "message": str(e),
                    "type": type(e).__name__,
                },
            )

    def _parse_input(self, input: dict[str, Any]) -> GenerationRequest:
        """Parse and validate input into GenerationRequest."""
        # Handle workflow enum
        workflow_str = input.get("workflow", "txt2img")
        workflow = WorkflowType(workflow_str)

        # Set default denoise based on workflow
        denoise = input.get("denoise")
        if denoise is None:
            if workflow == WorkflowType.IMG2IMG:
                denoise = 0.75
            elif workflow == WorkflowType.UPSCALE:
                denoise = 0.5
            else:
                denoise = 1.0

        # Default width/height for SDXL
        default_width = 1024 if workflow == WorkflowType.SDXL else 512
        default_height = 1024 if workflow == WorkflowType.SDXL else 512

        return GenerationRequest(
            prompt=input.get("prompt", ""),
            negative_prompt=input.get(
                "negative_prompt", "bad quality, blurry, distorted"
            ),
            width=input.get("width", default_width),
            height=input.get("height", default_height),
            steps=input.get("steps", 25 if workflow == WorkflowType.SDXL else 20),
            cfg_scale=input.get("cfg_scale", 7.0),
            seed=input.get("seed"),
            sampler=input.get("sampler", "euler"),
            scheduler=input.get("scheduler", "normal"),
            model=input.get("model"),
            workflow=workflow,
            denoise=denoise,
            input_image=input.get("input_image"),
            output_format=input.get("output_format", self.output_format),
            # LoRA parameters
            lora_name=input.get("lora_name"),
            lora_strength=input.get("lora_strength", 1.0),
            lora_clip_strength=input.get("lora_clip_strength", 1.0),
            # SDXL parameters
            refiner_model=input.get("refiner_model"),
            refiner_start=input.get("refiner_start", 0.8),
            # IP-Adapter parameters
            reference_image=input.get("reference_image"),
            ip_adapter_model=input.get(
                "ip_adapter_model", "ip-adapter_sd15.safetensors"
            ),
            clip_vision_model=input.get(
                "clip_vision_model", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            ),
            ip_adapter_weight=input.get("ip_adapter_weight", 1.0),
        )

    async def _resolve_model(self, requested_model: str | None) -> str | None:
        """Resolve which model to use."""
        # Use requested model if provided
        if requested_model:
            return requested_model

        # Use configured default
        if self.default_model:
            return self.default_model

        # Auto-detect first available model
        models = await self._client.get_available_models()
        if models:
            return models[0]

        return None

    async def _upload_input_image(self, image_base64: str) -> str:
        """Upload base64 image and return server filename."""
        # Decode base64
        image_data = base64.b64decode(image_base64)
        # Upload to ComfyUI
        return await self._client.upload_image(image_data, "input.png")

    def _build_workflow(
        self,
        request: GenerationRequest,
        model: str,
        input_image_name: str | None,
        reference_image_name: str | None = None,
    ) -> dict[str, Any]:
        """Build the appropriate workflow based on request."""
        common_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "seed": request.seed,
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            "model": model,
        }

        # Add LoRA params if specified
        lora_params = {}
        if request.lora_name:
            lora_params = {
                "lora_name": request.lora_name,
                "lora_strength": request.lora_strength,
                "lora_clip_strength": request.lora_clip_strength,
            }

        if request.workflow == WorkflowType.TXT2IMG:
            # Use LoRA workflow if LoRA specified, otherwise standard
            if request.lora_name:
                return self._txt2img_lora.build(
                    width=request.width,
                    height=request.height,
                    **common_params,
                    **lora_params,
                )
            return self._txt2img.build(
                width=request.width,
                height=request.height,
                **common_params,
            )
        elif request.workflow == WorkflowType.IMG2IMG:
            assert input_image_name is not None
            return self._img2img.build(
                input_image_name=input_image_name,
                width=request.width,
                height=request.height,
                denoise=request.denoise,
                **common_params,
            )
        elif request.workflow == WorkflowType.UPSCALE:
            assert input_image_name is not None
            return self._upscale.build(
                input_image_name=input_image_name,
                denoise=request.denoise,
                **common_params,
            )
        elif request.workflow == WorkflowType.SDXL:
            return self._sdxl.build(
                width=request.width,
                height=request.height,
                refiner_model=request.refiner_model,
                refiner_start=request.refiner_start,
                **common_params,
                **lora_params,
            )
        elif request.workflow == WorkflowType.IP_ADAPTER:
            assert reference_image_name is not None
            return self._ip_adapter.build(
                width=request.width,
                height=request.height,
                reference_image=reference_image_name,
                ip_adapter_model=request.ip_adapter_model,
                clip_vision_model=request.clip_vision_model,
                ip_adapter_weight=request.ip_adapter_weight,
                **common_params,
                **lora_params,
            )
        else:
            raise ValueError(f"Unsupported workflow type: {request.workflow}")

    async def get_status(self) -> dict[str, Any]:
        """Get ComfyUI server status."""
        status = await self._client.get_status()
        return {
            "online": status.online,
            "queue_remaining": status.queue_remaining,
            "queue_running": status.queue_running,
            "models_available": status.models_available,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self._client.close()
