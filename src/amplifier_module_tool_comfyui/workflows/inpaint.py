"""Inpainting workflow for masked image generation."""

import random
from typing import Any


class InpaintWorkflow:
    """Workflow for inpainting - generating content within masked areas.
    
    This workflow uses the SD 1.5 inpainting model to fill masked regions
    of an image while preserving unmasked areas. Useful for:
    - Generating around cutout areas (buttons, holes)
    - Replacing specific regions of an image
    - Extending images (outpainting)
    """

    def build(
        self,
        *,
        input_image_name: str,
        mask_image_name: str,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted",
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int | None = None,
        sampler: str = "euler",
        scheduler: str = "normal",
        model: str = "sd-v1-5-inpainting.ckpt",
        denoise: float = 1.0,
    ) -> dict[str, Any]:
        """Build the inpainting workflow.
        
        Args:
            input_image_name: Uploaded image filename
            mask_image_name: Uploaded mask filename (white = inpaint, black = preserve)
            prompt: What to generate in masked area
            negative_prompt: What to avoid
            steps: Sampling steps
            cfg_scale: Guidance scale
            seed: Random seed
            sampler: Sampling algorithm
            scheduler: Scheduler type
            model: Inpainting model name
            denoise: Denoising strength (1.0 = full regeneration in mask)
        
        Returns:
            ComfyUI workflow dict
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        return {
            # Load inpainting model
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model,
                },
            },
            # Load input image
            "2": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": input_image_name,
                },
            },
            # Load mask image
            "3": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": mask_image_name,
                },
            },
            # Convert mask to proper format (use alpha or convert to mask)
            "4": {
                "class_type": "ImageToMask",
                "inputs": {
                    "image": ["3", 0],
                    "channel": "red",  # Use red channel as mask
                },
            },
            # Encode positive prompt
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],
                },
            },
            # Encode negative prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
            },
            # VAE encode the image
            "7": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["2", 0],
                    "vae": ["1", 2],
                },
            },
            # Set the latent noise mask
            "8": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {
                    "samples": ["7", 0],
                    "mask": ["4", 0],
                },
            },
            # KSampler
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["8", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": denoise,
                },
            },
            # VAE decode
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["1", 2],
                },
            },
            # Save image
            "11": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["10", 0],
                    "filename_prefix": "inpaint",
                },
            },
        }
