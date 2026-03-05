# Next Steps: amplifier-module-tool-comfyui

*Generated 2026-03-04 — resume context for future sessions*

---

## Current State

The module is **substantially complete** but not production-ready. All 9 workflow types are implemented and the Amplifier tool contract is properly wired. The main gaps are test coverage, a few bugs, and some usability holes.

### What Works
- `generate_image` tool with full JSON schema covering all 9 workflows
- Async HTTP + WebSocket client with polling fallback
- All workflow builders: txt2img, img2img, upscale (latent), upscale_esrgan, inpaint, outpaint, controlnet, sdxl, ip_adapter
- Pydantic models for all request/response types
- Amplifier `mount()` entrypoint and `ToolResult` returns
- `docs/WORKFLOWS.md` and `examples/bundle.md` documentation

---

## Bugs to Fix

### 1. Dead code in `upscale.py` (Low effort)
Node `"12"` (`LatentUpscale`) is constructed at lines ~82-86 but never wired into the graph. Node `"13"` (`LatentUpscaleBy`) is what's actually used. The orphaned node is harmlessly ignored by ComfyUI but is confusing.

**Fix**: Remove or comment out the orphaned `LatentUpscale` node construction.

### 2. Inconsistent base class usage (Medium effort)
`ControlNetWorkflow`, `InpaintWorkflow`, and `OutpaintWorkflow` do not extend `WorkflowBuilder` — they use raw inline dicts instead of `create_node()`. The other 7 workflows properly subclass `WorkflowBuilder`.

**Fix**: Refactor the three outliers to extend `WorkflowBuilder` and use `create_node()`.

### 3. IP-Adapter undocumented custom node requirement (Low effort — docs only)
`IPAdapterModelLoader` and `IPAdapterApply` are not built-in ComfyUI nodes. They require the [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) custom node pack. Without it, the workflow silently fails.

**Fix**: Add a "Custom Nodes Required" section to the README listing:
- `ComfyUI_IPAdapter_plus` — required for `ip_adapter` workflow

### 4. Outpaint API mismatch (Medium effort)
The planning doc (`cabinet-art-module-plan.md`) described `extend_pixels` and `extend_direction` parameters for outpainting. These don't exist in `GenerationRequest` or the tool schema. The `outpaint` workflow is structurally identical to `inpaint` — callers must manually pre-pad the canvas and create the mask themselves.

**Fix options (pick one)**:
- A) Add `extend_pixels` + `extend_direction` to `GenerationRequest` and implement canvas padding + mask generation inside `OutpaintWorkflow.build()`
- B) Document the limitation clearly: update WORKFLOWS.md to explain callers must pre-pad, and note it in the tool schema description for `outpaint`

Option A is more useful but more work. Option B is a quick doc fix.

---

## Missing: Test Coverage

`tests/` is empty. Priority order for writing tests:

1. **Workflow builders** (pure unit tests, no server needed)
   - Each `WorkflowBuilder.build()` returns a dict
   - Assert expected node keys, connection wiring, parameter values
   - Fast, no mocking needed

2. **`ComfyUITool._parse_input()` and `_build_workflow()`**
   - Unit test input parsing and workflow dispatch logic
   - Mock the client to avoid HTTP

3. **`ComfyUIClient`**
   - Mock `httpx.AsyncClient` for REST calls
   - Mock `websockets.connect` for WebSocket path
   - Test error paths (timeout, execution_error, server offline)

Suggested tooling: `pytest` + `pytest-asyncio` + `respx` (httpx mock library).

---

## Open Design Questions (from cabinet-art-module-plan.md)

### Q1: Is 4× ESRGAN sufficient for print targets?
Target print spec: 24" × 52" @ 300 DPI = 7,200 × 15,600 px

If generating at 512×512 base, 4× ESRGAN gets to ~2048×2048 — not enough. If generating at 1024×1024 (SDXL), 4× gets to ~4096×4096 — still not enough for the largest sizes.

**Options**:
- Two-pass ESRGAN (4× twice = 16×): complex, quality degrades
- Tiled Diffusion: generate/refine in tiles, stitch — best quality but requires custom ComfyUI nodes (MultiDiffusion extension)
- Accept lower base res and rely on print shop upscaling for final output

### Q2: Tiled Diffusion (Phase 5 from plan — not implemented)
Needed for very high-res generation without quality loss from simple upscaling. Requires the [ComfyUI-MultiDiffusion-upscaler](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) or similar custom node pack.

Complexity: High. Recommended only if Q1 answer is "ESRGAN is not enough."

### Q3: Multi-ControlNet
Combining shape + depth ControlNet conditioning in a single pass. Not implemented — current ControlNet workflow supports one control image/type. Would require chaining `ControlNetApply` nodes.

### Q4: SDXL variants for inpaint/ControlNet
Current inpaint and ControlNet workflows are SD 1.5 only. SDXL inpainting requires a different model (SDXL-inpaint) and different node configuration.

---

## Feature Ideas Not Yet Planned

- **ControlNet preprocessors as tool calls**: Expose Canny edge detection, depth map extraction, etc. as separate `preprocess_image` tool actions so agents can prepare control images themselves.
- **Batch generation**: Run the same prompt N times with different seeds, return all results.
- **Workflow chaining**: txt2img → img2img → upscale in a single tool call as a convenience shortcut.
- **LoRA discovery**: `get_available_loras()` alongside the existing `get_available_models()`.

---

## Server Setup Notes

ComfyUI runs on the DGX Spark (GB10 GPU). The Docker launch script is at `/home/robotdad/comfy/start_comfyui_docker.sh` using:

```
nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

- ComfyUI path: `/home/robotdad/image-gen/ComfyUI`
- Model path: `/home/robotdad/image-gen/ComfyUI/models/checkpoints/`
- Default model: `v1-5-pruned-emaonly.safetensors`
- Port: 8188

Check server: `curl http://localhost:8188/system_stats`

IP-Adapter requires `ComfyUI_IPAdapter_plus` custom node pack installed in ComfyUI's `custom_nodes/` directory.
