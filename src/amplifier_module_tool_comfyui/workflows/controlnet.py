"""ControlNet workflow for structure-guided image generation."""

import random
from typing import Any


class ControlNetWorkflow:
    """Workflow for ControlNet-guided image generation.
    
    ControlNet allows using a control image (edges, depth, lines) to guide
    the structure of the generated image while the prompt controls content.
    
    For cabinet artwork: Use with edge maps of cabinet shapes to generate
    art that respects button cutouts, trim lines, and panel boundaries.
    
    Supported control types:
    - canny: Edge detection (best for sharp boundaries like cabinet outlines)
    - depth: Depth maps (for 3D spatial relationships)
    - lineart: Clean line drawings (for detailed structural control)
    """

    # Map control types to model filenames
    CONTROL_MODELS = {
        "canny": "control_v11p_sd15_canny.pth",
        "depth": "control_v11f1p_sd15_depth.pth",
        "lineart": "control_v11p_sd15_lineart.pth",
    }

    def build(
        self,
        *,
        control_image_name: str,
        control_type: str = "canny",
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
        controlnet_strength: float = 1.0,
    ) -> dict[str, Any]:
        """Build the ControlNet workflow.
        
        Args:
            control_image_name: Uploaded control image filename (edges/depth/lines)
            control_type: Type of control ("canny", "depth", "lineart")
            prompt: What to generate
            negative_prompt: What to avoid
            width: Output width
            height: Output height
            steps: Sampling steps
            cfg_scale: Guidance scale
            seed: Random seed
            sampler: Sampling algorithm
            scheduler: Scheduler type
            model: Base checkpoint model
            controlnet_strength: How strongly to follow control image (0-2)
        
        Returns:
            ComfyUI workflow dict
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        controlnet_model = self.CONTROL_MODELS.get(control_type, self.CONTROL_MODELS["canny"])

        return {
            # Load base checkpoint
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model,
                },
            },
            # Load ControlNet model
            "2": {
                "class_type": "ControlNetLoader",
                "inputs": {
                    "control_net_name": controlnet_model,
                },
            },
            # Load control image
            "3": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": control_image_name,
                },
            },
            # Encode positive prompt
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],
                },
            },
            # Encode negative prompt
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
            },
            # Apply ControlNet conditioning
            "6": {
                "class_type": "ControlNetApply",
                "inputs": {
                    "conditioning": ["4", 0],
                    "control_net": ["2", 0],
                    "image": ["3", 0],
                    "strength": controlnet_strength,
                },
            },
            # Empty latent for generation
            "7": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            # KSampler
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["6", 0],
                    "negative": ["5", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            },
            # VAE decode
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["1", 2],
                },
            },
            # Save image
            "10": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["9", 0],
                    "filename_prefix": f"controlnet_{control_type}",
                },
            },
        }
