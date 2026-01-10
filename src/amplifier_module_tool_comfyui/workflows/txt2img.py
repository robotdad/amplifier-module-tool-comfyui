"""Text-to-image workflow for ComfyUI."""

from typing import Any

from .base import WorkflowBuilder


class Txt2ImgWorkflow(WorkflowBuilder):
    """Standard text-to-image workflow using SD 1.5 / SDXL style models."""

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build text-to-image workflow.

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

        Returns:
            ComfyUI workflow dictionary
        """
        actual_seed = self.get_seed(seed)

        return {
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
            # Encode positive prompt
            "6": self.create_node(
                "CLIPTextEncode",
                {
                    "text": prompt,
                    "clip": ["4", 1],
                },
            ),
            # Encode negative prompt
            "7": self.create_node(
                "CLIPTextEncode",
                {
                    "text": negative_prompt,
                    "clip": ["4", 1],
                },
            ),
            # KSampler - the main sampling node
            "3": self.create_node(
                "KSampler",
                {
                    "model": ["4", 0],
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
            ),
            # Decode latent to image
            "8": self.create_node(
                "VAEDecode",
                {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            ),
            # Save image
            "9": self.create_node(
                "SaveImage",
                {
                    "images": ["8", 0],
                    "filename_prefix": "amplifier",
                },
            ),
        }
