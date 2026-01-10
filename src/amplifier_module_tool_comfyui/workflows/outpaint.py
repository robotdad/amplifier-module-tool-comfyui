"""Outpainting workflow for extending images beyond their borders."""

import random
from typing import Any


class OutpaintWorkflow:
    """Workflow for outpainting - extending images beyond their original borders.
    
    This workflow uses the inpainting model to generate content in areas
    beyond the original image. It pads the image and creates a mask for
    the new areas.
    
    For cabinet artwork: Use this to extend images to fit larger print
    dimensions while maintaining visual coherence.
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
        """Build the outpainting workflow.
        
        Note: Outpainting uses the same workflow as inpainting. The difference
        is in how the input image and mask are prepared:
        - Input image: Original image padded to the target size
        - Mask: White in the padded areas (to generate), black in original area
        
        The image padding and mask creation should be done before calling this
        workflow, either by the tool or by the calling application.
        
        Args:
            input_image_name: Uploaded padded image filename
            mask_image_name: Uploaded mask filename (white = generate, black = preserve)
            prompt: What to generate in the extended areas
            negative_prompt: What to avoid
            steps: Sampling steps
            cfg_scale: Guidance scale
            seed: Random seed
            sampler: Sampling algorithm
            scheduler: Scheduler type
            model: Inpainting model name
            denoise: Denoising strength
        
        Returns:
            ComfyUI workflow dict (same as inpainting)
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Outpainting uses the same workflow structure as inpainting
        return {
            # Load inpainting model
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model,
                },
            },
            # Load padded input image
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
            # Convert mask to proper format
            "4": {
                "class_type": "ImageToMask",
                "inputs": {
                    "image": ["3", 0],
                    "channel": "red",
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
                    "filename_prefix": "outpaint",
                },
            },
        }
