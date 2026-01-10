"""Upscale workflow for ComfyUI."""

from typing import Any

from .base import WorkflowBuilder


class UpscaleWorkflow(WorkflowBuilder):
    """Upscale workflow using latent upscaling with optional refinement."""

    def build(  # type: ignore[override]
        self,
        *,
        input_image_name: str,
        prompt: str = "",
        negative_prompt: str = "bad quality, blurry, distorted",
        scale_factor: float = 2.0,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int | None = None,
        sampler: str = "euler",
        scheduler: str = "normal",
        model: str = "v1-5-pruned-emaonly.safetensors",
        denoise: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build upscale workflow.

        This workflow:
        1. Loads the input image
        2. Encodes it to latent space
        3. Upscales the latent
        4. Refines with KSampler at low denoise
        5. Decodes back to image

        Args:
            input_image_name: Name of uploaded input image
            prompt: Optional prompt for guided upscaling
            negative_prompt: What to avoid
            scale_factor: Upscale multiplier (e.g., 2.0 for 2x)
            steps: Number of refinement steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (None for random)
            sampler: Sampler algorithm name
            scheduler: Scheduler type
            model: Checkpoint model filename
            denoise: Denoising strength for refinement (lower = preserve more detail)

        Returns:
            ComfyUI workflow dictionary
        """
        actual_seed = self.get_seed(seed)

        # Default prompt if not provided
        if not prompt:
            prompt = "high resolution, detailed, sharp"

        return {
            # Load checkpoint model
            "4": self.create_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": model},
            ),
            # Load input image
            "10": self.create_node(
                "LoadImage",
                {"image": input_image_name},
            ),
            # Encode image to latent
            "5": self.create_node(
                "VAEEncode",
                {
                    "pixels": ["10", 0],
                    "vae": ["4", 2],
                },
            ),
            # Upscale latent
            "12": self.create_node(
                "LatentUpscale",
                {
                    "samples": ["5", 0],
                    "upscale_method": "nearest-exact",
                    "width": 0,  # Will be calculated from scale
                    "height": 0,
                    "crop": "disabled",
                },
            ),
            # Alternatively use LatentUpscaleBy for factor-based scaling
            "13": self.create_node(
                "LatentUpscaleBy",
                {
                    "samples": ["5", 0],
                    "upscale_method": "bislerp",
                    "scale_by": scale_factor,
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
            # KSampler for refinement (low denoise preserves structure)
            "3": self.create_node(
                "KSampler",
                {
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["13", 0],  # Use factor-based upscale
                    "seed": actual_seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": denoise,
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
                    "filename_prefix": "amplifier_upscale",
                },
            ),
        }
