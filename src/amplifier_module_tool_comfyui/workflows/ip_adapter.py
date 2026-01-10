"""IP-Adapter workflow for ComfyUI - use reference images to guide generation."""

from typing import Any

from .base import WorkflowBuilder


class IPAdapterWorkflow(WorkflowBuilder):
    """IP-Adapter workflow for style/subject transfer from reference images."""

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
        reference_image: str = "",
        ip_adapter_model: str = "ip-adapter_sd15.safetensors",
        clip_vision_model: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        ip_adapter_weight: float = 1.0,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
        lora_clip_strength: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build IP-Adapter workflow for reference-guided generation.

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
            reference_image: Server filename of uploaded reference image
            ip_adapter_model: IP-Adapter model name
            clip_vision_model: CLIP vision encoder model name
            ip_adapter_weight: Strength of IP-Adapter influence (0-2)
            lora_name: Optional LoRA model
            lora_strength: LoRA strength
            lora_clip_strength: LoRA CLIP strength

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
            # Create empty latent
            "5": self.create_node(
                "EmptyLatentImage",
                {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            ),
            # Load reference image
            "30": self.create_node(
                "LoadImage",
                {"image": reference_image},
            ),
            # Load CLIP Vision model
            "31": self.create_node(
                "CLIPVisionLoader",
                {"clip_name": clip_vision_model},
            ),
            # Load IP-Adapter model
            "32": self.create_node(
                "IPAdapterModelLoader",
                {"ipadapter_file": ip_adapter_model},
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

        # Apply IP-Adapter to model
        workflow["33"] = self.create_node(
            "IPAdapterApply",
            {
                "model": model_source,
                "ipadapter": ["32", 0],
                "clip_vision": ["31", 0],
                "image": ["30", 0],
                "weight": ip_adapter_weight,
                "noise": 0.0,
                "weight_type": "original",
                "start_at": 0.0,
                "end_at": 1.0,
            },
        )

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

        # KSampler with IP-Adapter enhanced model
        workflow["3"] = self.create_node(
            "KSampler",
            {
                "model": ["33", 0],  # Use IP-Adapter modified model
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
                "filename_prefix": "amplifier_ipadapter",
            },
        )

        return workflow
