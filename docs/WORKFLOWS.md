# ComfyUI Tool Workflows

This document provides detailed information about each workflow available in the ComfyUI tool module.

## Workflow Overview

| Workflow | Purpose | Required Models |
|----------|---------|-----------------|
| `txt2img` | Generate from text | SD checkpoint |
| `img2img` | Transform images | SD checkpoint |
| `upscale` | Diffusion upscale | SD checkpoint |
| `upscale_esrgan` | Fast AI upscale | ESRGAN model |
| `inpaint` | Fill masked areas | Inpainting checkpoint |
| `outpaint` | Extend image borders | Inpainting checkpoint |
| `controlnet` | Structure-guided gen | SD checkpoint + ControlNet |
| `sdxl` | High-res generation | SDXL checkpoint |
| `ip_adapter` | Style transfer | SD checkpoint + IP-Adapter |

## txt2img (Text-to-Image)

The default workflow. Generates images from text descriptions.

### Parameters
- `prompt` (required): What to generate
- `negative_prompt`: What to avoid
- `width`, `height`: Output dimensions (default 512x512)
- `steps`: Sampling steps (default 20)
- `cfg_scale`: Guidance strength (default 7.0)
- `seed`: For reproducibility
- `sampler`: Algorithm (default "euler")
- `model`: Checkpoint to use

### With LoRA
Add style customization with LoRA models:
- `lora_name`: LoRA filename
- `lora_strength`: Model influence (0-2)
- `lora_clip_strength`: CLIP influence (0-2)

## img2img (Image-to-Image)

Transform an existing image based on a prompt while preserving structure.

### Parameters
- `input_image` (required): Base64-encoded source image
- `denoise`: How much to change (0=none, 1=complete). Default 0.75
- All txt2img parameters

### Tips
- Lower denoise (0.3-0.5) preserves more of the original
- Higher denoise (0.7-0.9) allows more creative freedom

## upscale (Diffusion Upscale)

Upscale images using diffusion to add detail. Slower but can enhance quality.

### Parameters
- `input_image` (required): Base64-encoded image to upscale
- `denoise`: Detail enhancement level (default 0.5)
- All txt2img parameters (prompt guides added details)

### When to Use
- When you want AI-enhanced details during upscale
- When source image quality is lower

## upscale_esrgan (Real-ESRGAN Upscale)

Fast 4x upscaling using Real-ESRGAN. No diffusion, no prompt needed.

### Parameters
- `input_image` (required): Base64-encoded image
- `upscaler_model`: Model to use
  - `RealESRGAN_x4plus.pth` - General purpose (default)
  - `RealESRGAN_x4plus_anime_6B.pth` - Anime/illustration

### When to Use
- Final step before print (fast, deterministic)
- When you need exact 4x scaling
- When source is already high quality

## inpaint (Inpainting)

Fill masked areas of an image with generated content.

### Parameters
- `input_image` (required): Base64-encoded source image
- `mask_image` (required): Base64-encoded mask
  - **White** = area to regenerate
  - **Black** = area to preserve
- `model`: Use inpainting-specific model (`sd-v1-5-inpainting.ckpt`)
- `denoise`: Generation strength (default 1.0)

### Use Cases
- Removing objects from images
- Filling cutout areas (e.g., cabinet button holes)
- Replacing specific regions

## outpaint (Outpainting)

Extend images beyond their original borders.

### Parameters
Same as inpaint, but input image should be:
1. **Pre-padded** to target dimensions
2. **Mask** covers the new padded areas (white)

### Workflow
1. Pad original image to larger canvas (e.g., with gray)
2. Create mask: white for new areas, black for original
3. Run outpaint with descriptive prompt

### Use Cases
- Extending artwork to fit larger print sizes
- Adding more sky/ground to landscapes
- Expanding portraits to include more background

## controlnet (ControlNet)

Generate images guided by a control image (edges, depth, lines).

### Parameters
- `control_image` (required): Base64-encoded control image
- `control_type`: Type of control signal
  - `canny` - Edge detection (sharp boundaries)
  - `depth` - Depth maps (3D relationships)
  - `lineart` - Line drawings (structural detail)
- `controlnet_strength`: How strictly to follow control (0-2, default 1.0)
- All txt2img parameters

### Creating Control Images

**Canny (edges)**:
- Use edge detection on your reference
- White lines on black background
- Best for architectural/mechanical shapes

**Depth**:
- Grayscale depth map (closer=lighter)
- Good for scenes with spatial depth

**Lineart**:
- Clean line drawings
- Good for detailed illustrations

### Use Cases
- Generate art matching cabinet shapes/cutouts
- Create variations while maintaining structure
- Convert sketches to finished art

## sdxl (SDXL)

High-quality generation at 1024x1024 using SDXL models.

### Parameters
- `model`: SDXL base model
- `refiner_model`: Optional SDXL refiner
- `refiner_start`: When to switch to refiner (0-1, default 0.8)
- Default dimensions: 1024x1024
- Default steps: 25

### With Refiner
The refiner adds fine details in the final steps:
- `refiner_start: 0.8` means base runs 80%, refiner runs 20%
- Lower values = more refiner influence

## ip_adapter (IP-Adapter)

Use a reference image to guide generation style/content.

### Parameters
- `reference_image` (required): Base64-encoded style reference
- `ip_adapter_model`: IP-Adapter model name
- `clip_vision_model`: CLIP vision encoder
- `ip_adapter_weight`: Reference influence (0-2, default 1.0)
- All txt2img parameters

### Use Cases
- Generate images in the style of a reference
- Create variations of a concept
- Style transfer while varying content

## Combining Workflows

For complex tasks, chain workflows:

### Example: Print-Ready Cabinet Art

1. **ControlNet**: Generate base art respecting cabinet shape
2. **Outpaint**: Extend to full print dimensions
3. **Upscale ESRGAN**: 4x upscale for 300dpi

### Example: Style-Consistent Series

1. **IP-Adapter**: Generate first image
2. Use that as reference for subsequent generations
3. Consistent style across the series
