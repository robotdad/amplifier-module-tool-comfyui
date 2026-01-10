# amplifier-module-tool-comfyui

Amplifier tool module for AI-powered image generation via [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Features

- **Multiple Workflows**: txt2img, img2img, upscale, upscale_esrgan, inpaint, outpaint, controlnet, sdxl, ip_adapter
- **LoRA Support**: Apply LoRA models to customize generation style
- **ControlNet**: Structure-guided generation using edge/depth/line control images
- **Inpainting/Outpainting**: Fill masked areas or extend images beyond borders
- **AI Upscaling**: Real-ESRGAN 4x upscaling for print-quality output
- **WebSocket Progress**: Real-time generation progress updates
- **Auto Model Detection**: Automatically uses available checkpoint models

## Requirements

- ComfyUI server running and accessible
- Python 3.11+
- Models installed based on workflows you plan to use (see [Model Requirements](#model-requirements))

## Installation

```bash
# From source
pip install git+https://github.com/robotdad/amplifier-module-tool-comfyui.git
```

## Configuration

Add to your Amplifier bundle or settings:

```yaml
tools:
  - module: comfyui
    source: git+https://github.com/robotdad/amplifier-module-tool-comfyui@main
    config:
      base_url: "http://127.0.0.1:8188"  # ComfyUI server URL
      timeout: 300                        # Timeout in seconds
      default_model: "v1-5-pruned-emaonly.safetensors"  # Optional
      output_format: "path"               # "path" or "base64"
      use_websocket: true                 # Use WebSocket for progress
```

## Workflows

### txt2img (default)
Generate images from text descriptions.

```json
{
  "prompt": "a majestic mountain at sunset, oil painting style",
  "negative_prompt": "blurry, low quality",
  "width": 768,
  "height": 512,
  "steps": 25,
  "cfg_scale": 7.5
}
```

### img2img
Transform existing images based on prompts.

```json
{
  "workflow": "img2img",
  "prompt": "convert to watercolor painting style",
  "input_image": "<base64-encoded-image>",
  "denoise": 0.6
}
```

### upscale
Upscale images with diffusion-based refinement (slower, adds detail).

```json
{
  "workflow": "upscale",
  "input_image": "<base64-encoded-image>",
  "prompt": "high resolution, detailed",
  "denoise": 0.4
}
```

### upscale_esrgan
Fast AI upscaling using Real-ESRGAN (4x, no diffusion). Best for print preparation.

```json
{
  "workflow": "upscale_esrgan",
  "input_image": "<base64-encoded-image>",
  "upscaler_model": "RealESRGAN_x4plus.pth"
}
```

### inpaint
Fill masked areas of an image. Mask should be white where you want to generate, black where you want to preserve.

```json
{
  "workflow": "inpaint",
  "prompt": "a red flower",
  "input_image": "<base64-encoded-image>",
  "mask_image": "<base64-encoded-mask>",
  "model": "sd-v1-5-inpainting.ckpt"
}
```

### outpaint
Extend images beyond their borders. Similar to inpaint but for expanding canvas. Pre-pad your image and create a mask where new content should be generated.

```json
{
  "workflow": "outpaint",
  "prompt": "seamless landscape extension, matching style",
  "input_image": "<base64-padded-image>",
  "mask_image": "<base64-mask-for-new-areas>",
  "model": "sd-v1-5-inpainting.ckpt"
}
```

### controlnet
Structure-guided generation using control images (edges, depth, lines).

```json
{
  "workflow": "controlnet",
  "prompt": "a colorful retro arcade cabinet",
  "control_image": "<base64-edge-map>",
  "control_type": "canny",
  "controlnet_strength": 0.8,
  "width": 512,
  "height": 512
}
```

Control types:
- `canny` - Edge detection (best for sharp boundaries)
- `depth` - Depth maps (for 3D spatial relationships)
- `lineart` - Clean line drawings (for detailed structural control)

### sdxl
High-quality generation at 1024x1024 using SDXL models, with optional refiner.

```json
{
  "workflow": "sdxl",
  "prompt": "a photorealistic portrait",
  "width": 1024,
  "height": 1024,
  "model": "sd_xl_base_1.0.safetensors",
  "refiner_model": "sd_xl_refiner_1.0.safetensors",
  "refiner_start": 0.8
}
```

### ip_adapter
Use a reference image to guide the style/content of generation.

```json
{
  "workflow": "ip_adapter",
  "prompt": "a portrait in the same style",
  "reference_image": "<base64-reference>",
  "ip_adapter_weight": 0.8
}
```

## LoRA Support

Add LoRA models to txt2img or sdxl workflows:

```json
{
  "prompt": "a character portrait",
  "lora_name": "my_style_lora.safetensors",
  "lora_strength": 0.8,
  "lora_clip_strength": 0.8
}
```

## Parameters Reference

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | "bad quality, blurry, distorted" | What to avoid |
| `width` | int | 512 | Image width (64-2048) |
| `height` | int | 512 | Image height (64-2048) |
| `steps` | int | 20 | Sampling steps (1-150) |
| `cfg_scale` | float | 7.0 | Guidance scale (1-30) |
| `seed` | int | random | Random seed for reproducibility |
| `sampler` | string | "euler" | Sampling algorithm |
| `scheduler` | string | "normal" | Scheduler type |
| `model` | string | auto | Checkpoint model name |
| `workflow` | string | "txt2img" | Workflow type |
| `output_format` | string | "path" | "path" or "base64" |

### Workflow-Specific Parameters

| Parameter | Workflows | Description |
|-----------|-----------|-------------|
| `input_image` | img2img, upscale, upscale_esrgan, inpaint, outpaint | Base64 input image |
| `mask_image` | inpaint, outpaint | Base64 mask (white=generate) |
| `denoise` | img2img, upscale, inpaint, outpaint | Denoising strength (0-1) |
| `upscaler_model` | upscale_esrgan | ESRGAN model name |
| `control_image` | controlnet | Base64 control image |
| `control_type` | controlnet | "canny", "depth", or "lineart" |
| `controlnet_strength` | controlnet | Control influence (0-2) |
| `reference_image` | ip_adapter | Base64 reference image |
| `ip_adapter_weight` | ip_adapter | Reference influence (0-2) |
| `refiner_model` | sdxl | SDXL refiner model |
| `refiner_start` | sdxl | When to switch to refiner (0-1) |
| `lora_name` | txt2img, sdxl | LoRA model filename |
| `lora_strength` | txt2img, sdxl | LoRA model strength (0-2) |

## Model Requirements

### Base Models (checkpoints/)
- **SD 1.5**: `v1-5-pruned-emaonly.safetensors` - General purpose
- **SD 1.5 Inpainting**: `sd-v1-5-inpainting.ckpt` - For inpaint/outpaint
- **SDXL Base**: `sd_xl_base_1.0.safetensors` - High quality 1024px
- **SDXL Refiner**: `sd_xl_refiner_1.0.safetensors` - Optional refinement

### Upscaler Models (upscale_models/)
- `RealESRGAN_x4plus.pth` - General purpose 4x upscaling
- `RealESRGAN_x4plus_anime_6B.pth` - Anime/illustration focused

### ControlNet Models (controlnet/)
- `control_v11p_sd15_canny.pth` - Edge detection control
- `control_v11f1p_sd15_depth.pth` - Depth map control
- `control_v11p_sd15_lineart.pth` - Line art control

### IP-Adapter Models (ipadapter/)
- `ip-adapter_sd15.safetensors`

### CLIP Vision (clip_vision/)
- `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`

## ComfyUI Setup

1. Install ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. Download models to appropriate folders under `ComfyUI/models/`

3. Start the server:
   ```bash
   python main.py --listen 0.0.0.0 --port 8188
   ```

## Example Use Case: Cabinet Artwork

For generating arcade cabinet artwork with specific shapes/cutouts:

1. **Create edge map** of cabinet shape (cutouts, trim lines, panel boundaries)
2. **Generate with ControlNet** using the edge map to respect boundaries
3. **Outpaint** to extend to full print dimensions (e.g., 24"x52")
4. **Upscale with ESRGAN** for 300dpi print quality

## Development

```bash
# Clone repository
git clone https://github.com/robotdad/amplifier-module-tool-comfyui.git
cd amplifier-module-tool-comfyui

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
