"""Image-to-image workflow for ComfyUI."""

from typing import Any

from .base import WorkflowBuilder


class Img2ImgWorkflow(WorkflowBuilder):
    """Image-to-image workflow that transforms an input image based on a prompt."""

    def build(  # type: ignore[override]
        self,
        *,
        prompt: str,
        input_image_name: str,
        negative_prompt: str = "bad quality, blurry, distorted",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int | None = None,
        sampler: str = "euler",
        scheduler: str = "normal",
        model: str = "v1-5-pruned-emaonly.safetensors",
        denoise: float = 0.75,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build image-to-image workflow.

        Args:
            prompt: Positive prompt describing desired transformation
            input_image_name: Name of uploaded input image
            negative_prompt: What to avoid in the image
            width: Output image width in pixels
            height: Output image height in pixels
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (None for random)
            sampler: Sampler algorithm name
            scheduler: Scheduler type
            model: Checkpoint model filename
            denoise: Denoising strength (0.0-1.0, higher = more change)

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
            # Load input image
            "10": self.create_node(
                "LoadImage",
                {"image": input_image_name},
            ),
            # Resize image if needed
            "11": self.create_node(
                "ImageScale",
                {
                    "image": ["10", 0],
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "crop": "center",
                },
            ),
            # Encode image to latent
            "5": self.create_node(
                "VAEEncode",
                {
                    "pixels": ["11", 0],
                    "vae": ["4", 2],
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
            # KSampler with denoise < 1.0 for img2img
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
                    "filename_prefix": "amplifier_img2img",
                },
            ),
        }
