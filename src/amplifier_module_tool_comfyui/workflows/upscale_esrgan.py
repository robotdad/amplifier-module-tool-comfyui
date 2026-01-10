"""Real-ESRGAN upscaler workflow for print-quality upscaling."""

from typing import Any


class UpscaleESRGANWorkflow:
    """Workflow for AI-based image upscaling using Real-ESRGAN models.
    
    This workflow uses dedicated upscaler models (not latent upscaling) for
    high-quality 4x upscaling suitable for print output.
    """

    def build(
        self,
        *,
        model_name: str = "RealESRGAN_x4plus.pth",
    ) -> dict[str, Any]:
        """Build the upscaler workflow.
        
        Args:
            model_name: Upscaler model filename (in models/upscale_models/)
        
        Returns:
            ComfyUI workflow dict. Note: image is loaded separately and
            connected by the tool before execution.
        """
        return {
            # Load the input image (will be connected by tool)
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": "",  # Filled by tool with uploaded image name
                },
            },
            # Load upscaler model
            "2": {
                "class_type": "UpscaleModelLoader",
                "inputs": {
                    "model_name": model_name,
                },
            },
            # Apply upscaler
            "3": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image": ["1", 0],
                },
            },
            # Save output
            "4": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["3", 0],
                    "filename_prefix": "upscaled",
                },
            },
        }
