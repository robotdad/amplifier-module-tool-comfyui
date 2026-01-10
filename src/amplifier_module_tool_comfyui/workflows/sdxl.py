"""SDXL workflow for ComfyUI - higher quality generation at 1024x1024."""

from typing import Any

from .base import WorkflowBuilder


class SDXLWorkflow(WorkflowBuilder):
    """SDXL text-to-image workflow with optional refiner pass."""

    def build(  # type: ignore[override]
        self,
        *,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        cfg_scale: float = 7.0,
        seed: int | None = None,
        sampler: str = "euler",
        scheduler: str = "normal",
        model: str = "sd_xl_base_1.0.safetensors",
        refiner_model: str | None = None,
        refiner_start: float = 0.8,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
        lora_clip_strength: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build SDXL workflow with optional refiner.

        Args:
            prompt: Positive prompt describing desired image
            negative_prompt: What to avoid in the image
            width: Image width (default 1024 for SDXL)
            height: Image height (default 1024 for SDXL)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (None for random)
            sampler: Sampler algorithm name
            scheduler: Scheduler type
            model: SDXL base model filename
            refiner_model: SDXL refiner model (optional)
            refiner_start: When to switch to refiner (0-1, default 0.8)
            lora_name: LoRA model filename (optional)
            lora_strength: LoRA strength (0-2)
            lora_clip_strength: LoRA CLIP strength (0-2)

        Returns:
            ComfyUI workflow dictionary
        """
        actual_seed = self.get_seed(seed)

        workflow: dict[str, Any] = {
            # Load SDXL base model
            "4": self.create_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": model},
            ),
            # Create empty latent (SDXL uses same latent format)
            "5": self.create_node(
                "EmptyLatentImage",
                {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            ),
        }

        model_source = ["4", 0]
        clip_source = ["4", 1]

        # Add LoRA if specified
        if lora_name:
            workflow["20"] = self.create_node(
                "LoraLoader",
                {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_clip_strength,
                },
            )
            model_source = ["20", 0]
            clip_source = ["20", 1]

        # Encode positive prompt
        workflow["6"] = self.create_node(
            "CLIPTextEncode",
            {
                "text": prompt,
                "clip": clip_source,
            },
        )
        # Encode negative prompt
        workflow["7"] = self.create_node(
            "CLIPTextEncode",
            {
                "text": negative_prompt,
                "clip": clip_source,
            },
        )

        if refiner_model:
            # Two-pass workflow with refiner
            base_steps = int(steps * refiner_start)
            refiner_steps = steps - base_steps

            # Load refiner model
            workflow["14"] = self.create_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": refiner_model},
            )

            # Base model pass
            workflow["3"] = self.create_node(
                "KSampler",
                {
                    "model": model_source,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "seed": actual_seed,
                    "steps": base_steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            )

            # Encode prompts for refiner
            workflow["15"] = self.create_node(
                "CLIPTextEncode",
                {
                    "text": prompt,
                    "clip": ["14", 1],
                },
            )
            workflow["16"] = self.create_node(
                "CLIPTextEncode",
                {
                    "text": negative_prompt,
                    "clip": ["14", 1],
                },
            )

            # Refiner pass
            workflow["17"] = self.create_node(
                "KSampler",
                {
                    "model": ["14", 0],
                    "positive": ["15", 0],
                    "negative": ["16", 0],
                    "latent_image": ["3", 0],
                    "seed": actual_seed,
                    "steps": refiner_steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0 - refiner_start,
                },
            )

            # Decode from refiner output
            workflow["8"] = self.create_node(
                "VAEDecode",
                {
                    "samples": ["17", 0],
                    "vae": ["14", 2],
                },
            )
        else:
            # Single pass without refiner
            workflow["3"] = self.create_node(
                "KSampler",
                {
                    "model": model_source,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "seed": actual_seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            )
            workflow["8"] = self.create_node(
                "VAEDecode",
                {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            )

        # Save image
        workflow["9"] = self.create_node(
            "SaveImage",
            {
                "images": ["8", 0],
                "filename_prefix": "amplifier_sdxl",
            },
        )

        return workflow
