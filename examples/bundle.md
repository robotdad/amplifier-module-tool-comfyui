---
bundle:
  name: comfyui-example
  version: 1.0.0
  description: Example bundle demonstrating ComfyUI image generation tool

tools:
  - module: comfyui
    source: git+https://github.com/robotdad/amplifier-module-tool-comfyui@main
    config:
      base_url: "http://127.0.0.1:8188"
      timeout: 300
      output_format: "base64"
      use_websocket: true
---

# ComfyUI Image Generation

You have access to a powerful image generation tool via ComfyUI.

## Available Workflows

- **txt2img**: Generate images from text descriptions
- **img2img**: Transform existing images
- **upscale_esrgan**: Fast 4x AI upscaling (best for print)
- **inpaint**: Fill masked areas of images
- **outpaint**: Extend images beyond their borders
- **controlnet**: Structure-guided generation using edge/depth/line maps
- **sdxl**: High-quality 1024x1024 generation
- **ip_adapter**: Style transfer using reference images

## Quick Examples

### Text-to-Image
```json
{
  "prompt": "a vibrant retro arcade cabinet, neon lights, 80s aesthetic",
  "width": 768,
  "height": 1024,
  "steps": 25
}
```

### ControlNet (with edge map)
```json
{
  "workflow": "controlnet",
  "prompt": "colorful abstract art, geometric patterns",
  "control_image": "<base64-edge-map>",
  "control_type": "canny",
  "controlnet_strength": 0.8
}
```

### Inpainting
```json
{
  "workflow": "inpaint",
  "prompt": "seamless background texture",
  "input_image": "<base64-image>",
  "mask_image": "<base64-mask>",
  "model": "sd-v1-5-inpainting.ckpt"
}
```

### Fast Upscale for Print
```json
{
  "workflow": "upscale_esrgan",
  "input_image": "<base64-image>"
}
```

## Tips

1. **For cabinet artwork**: Use ControlNet with edge maps of your cabinet shape
2. **For print quality**: Always finish with upscale_esrgan for 4x resolution
3. **For extending images**: Use outpaint with a padded image and appropriate mask
4. **For consistency**: Use ip_adapter with a reference image
