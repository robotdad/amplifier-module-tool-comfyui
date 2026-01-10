"""Text-to-image workflow with LoRA support for ComfyUI."""

from typing import Any

from .base import WorkflowBuilder


class Txt2ImgLoRAWorkflow(WorkflowBuilder):
    """Text-to-image workflow with LoRA model injection."""

    def build(  # type: ignore[override]
        self,
        *,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int | None = None,
        sampler: str = "euler",
        scheduler: str = "normal",
        model: str = "v1-5-pruned-emaonly.safetensors",
        lora_name: str | None = None,
        lora_strength: float = 1.0,
        lora_clip_strength: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build text-to-image workflow with optional LoRA.

        Args:
            prompt: Positive prompt describing desired image
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (None for random)
            sampler: Sampler algorithm name
            scheduler: Scheduler type
            model: Checkpoint model filename
            lora_name: LoRA model filename (optional)
            lora_strength: LoRA model strength (0-2, default 1.0)
            lora_clip_strength: LoRA CLIP strength (0-2, default 1.0)

        Returns:
            ComfyUI workflow dictionary
        """
        actual_seed = self.get_seed(seed)

        workflow: dict[str, Any] = {
            # Load checkpoint model
            "4": self.create_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": model},
            ),
            # Create empty latent image
            "5": self.create_node(
                "EmptyLatentImage",
                {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            ),
        }

        # Model and CLIP source - either from checkpoint or LoRA
        model_source = ["4", 0]
        clip_source = ["4", 1]

        # Add LoRA loader if specified
        if lora_name:
            workflow["10"] = self.create_node(
                "LoraLoader",
                {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_clip_strength,
                },
            )
            model_source = ["10", 0]
            clip_source = ["10", 1]

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
        # KSampler - the main sampling node
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
        # Decode latent to image
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
                "filename_prefix": "amplifier",
            },
        )

        return workflow
