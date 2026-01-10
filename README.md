# amplifier-module-tool-comfyui

Amplifier tool module for AI-powered image generation via [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Features

- **Text-to-Image**: Generate images from text descriptions
- **Image-to-Image**: Transform existing images based on prompts
- **Upscaling**: Upscale images with optional AI refinement
- **WebSocket Progress**: Real-time generation progress updates
- **Auto Model Detection**: Automatically uses available checkpoint models

## Requirements

- ComfyUI server running and accessible
- At least one Stable Diffusion checkpoint model installed
- Python 3.11+

## Installation

```bash
# Via pip (when published)
pip install amplifier-module-tool-comfyui

# From source
pip install git+https://github.com/robotdad/amplifier-module-tool-comfyui.git
```

## Configuration

Add to your Amplifier mount plan:

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

## Usage Examples

### Text-to-Image

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

### Image-to-Image

```json
{
  "workflow": "img2img",
  "prompt": "convert to watercolor painting style",
  "input_image": "<base64-encoded-image>",
  "denoise": 0.6
}
```

### Upscale

```json
{
  "workflow": "upscale",
  "input_image": "<base64-encoded-image>",
  "prompt": "high resolution, detailed",
  "denoise": 0.4
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | "bad quality, blurry, distorted" | What to avoid |
| `width` | int | 512 | Image width (64-2048) |
| `height` | int | 512 | Image height (64-2048) |
| `steps` | int | 20 | Sampling steps (1-150) |
| `cfg_scale` | float | 7.0 | Guidance scale (1-30) |
| `seed` | int | random | Random seed for reproducibility |
| `workflow` | string | "txt2img" | "txt2img", "img2img", or "upscale" |
| `input_image` | string | - | Base64 image (required for img2img/upscale) |
| `denoise` | float | varies | Denoising strength (0-1) |
| `model` | string | auto | Checkpoint model name |

## ComfyUI Setup

1. Install ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. Download a model (e.g., SD 1.5):
   ```bash
   cd models/checkpoints
   wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
   ```

3. Start the server:
   ```bash
   python main.py --listen 0.0.0.0 --port 8188
   ```

## Development

```bash
# Clone repository
git clone https://github.com/robotdad/amplifier-module-tool-comfyui.git
cd amplifier-module-tool-comfyui

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
