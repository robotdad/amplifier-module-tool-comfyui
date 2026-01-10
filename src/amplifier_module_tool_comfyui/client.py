"""Async client for ComfyUI REST and WebSocket API."""

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncIterator

import httpx
import websockets
from websockets.asyncio.client import ClientConnection

from .models import ComfyUIStatus, GeneratedImage, GenerationProgress, GenerationResult


class ComfyUIClient:
    """Async client for interacting with ComfyUI server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self._http_client: httpx.AsyncClient | None = None

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL from base URL."""
        ws_base = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        return f"{ws_base}/ws?clientId={self.client_id}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def health_check(self) -> bool:
        """Check if ComfyUI server is responsive."""
        try:
            client = await self._get_client()
            response = await client.get("/system_stats")
            return response.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> ComfyUIStatus:
        """Get ComfyUI server status including available models."""
        try:
            client = await self._get_client()

            # Get system stats
            stats_response = await client.get("/system_stats")
            system_stats = (
                stats_response.json() if stats_response.status_code == 200 else {}
            )

            # Get queue status
            queue_response = await client.get("/queue")
            queue_data = (
                queue_response.json() if queue_response.status_code == 200 else {}
            )

            # Get available models (checkpoints)
            models = await self.get_available_models()

            return ComfyUIStatus(
                online=True,
                queue_remaining=len(queue_data.get("queue_pending", [])),
                queue_running=len(queue_data.get("queue_running", [])),
                models_available=models,
                system_stats=system_stats,
            )
        except Exception:
            return ComfyUIStatus(online=False)

    async def get_available_models(self) -> list[str]:
        """Get list of available checkpoint models."""
        try:
            client = await self._get_client()
            response = await client.get("/object_info/CheckpointLoaderSimple")
            if response.status_code == 200:
                data = response.json()
                # Navigate to the checkpoint list
                ckpt_info = data.get("CheckpointLoaderSimple", {})
                inputs = ckpt_info.get("input", {}).get("required", {})
                ckpt_name = inputs.get("ckpt_name", [])
                if isinstance(ckpt_name, list) and len(ckpt_name) > 0:
                    return ckpt_name[0] if isinstance(ckpt_name[0], list) else []
            return []
        except Exception:
            return []

    async def get_available_samplers(self) -> list[str]:
        """Get list of available samplers."""
        try:
            client = await self._get_client()
            response = await client.get("/object_info/KSampler")
            if response.status_code == 200:
                data = response.json()
                sampler_info = data.get("KSampler", {})
                inputs = sampler_info.get("input", {}).get("required", {})
                sampler_name = inputs.get("sampler_name", [])
                if isinstance(sampler_name, list) and len(sampler_name) > 0:
                    return sampler_name[0] if isinstance(sampler_name[0], list) else []
            return []
        except Exception:
            return []

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload an image to ComfyUI for img2img workflows."""
        client = await self._get_client()
        files = {"image": (filename, image_data, "image/png")}
        response = await client.post("/upload/image", files=files)
        response.raise_for_status()
        result = response.json()
        return result.get("name", filename)

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow prompt for execution. Returns prompt_id."""
        client = await self._get_client()
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        response = await client.post("/prompt", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Get execution history for a prompt."""
        client = await self._get_client()
        response = await client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    async def get_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> bytes:
        """Download a generated image."""
        client = await self._get_client()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = await client.get("/view", params=params)
        response.raise_for_status()
        return response.content

    async def wait_for_completion_polling(
        self,
        prompt_id: str,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> GenerationResult:
        """Wait for prompt completion using polling (simpler, no WebSocket)."""
        timeout = timeout or self.timeout
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Generation timed out after {timeout}s")

            history = await self.get_history(prompt_id)
            if prompt_id in history:
                prompt_history = history[prompt_id]
                if "outputs" in prompt_history:
                    return await self._parse_history_result(prompt_id, prompt_history)

            await asyncio.sleep(poll_interval)

    async def wait_for_completion_websocket(
        self,
        prompt_id: str,
        timeout: float | None = None,
    ) -> AsyncIterator[GenerationProgress | GenerationResult]:
        """Wait for completion using WebSocket with progress updates."""
        timeout = timeout or self.timeout

        async with websockets.connect(self.ws_url) as ws:
            async for message in self._process_websocket(ws, prompt_id, timeout):
                yield message

    async def _process_websocket(
        self,
        ws: ClientConnection,
        prompt_id: str,
        timeout: float,
    ) -> AsyncIterator[GenerationProgress | GenerationResult]:
        """Process WebSocket messages until completion."""
        try:
            async with asyncio.timeout(timeout):
                async for message in ws:
                    if isinstance(message, bytes):
                        # Binary message - preview image
                        continue

                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_data = data.get("data", {})

                    if msg_type == "progress":
                        yield GenerationProgress(
                            prompt_id=prompt_id,
                            step=msg_data.get("value"),
                            total_steps=msg_data.get("max"),
                        )
                    elif msg_type == "executing":
                        node = msg_data.get("node")
                        if node is None and msg_data.get("prompt_id") == prompt_id:
                            # Execution complete
                            history = await self.get_history(prompt_id)
                            if prompt_id in history:
                                yield await self._parse_history_result(
                                    prompt_id, history[prompt_id]
                                )
                            return
                        yield GenerationProgress(prompt_id=prompt_id, node=node)
                    elif msg_type == "execution_error":
                        error_msg = msg_data.get("exception_message", "Unknown error")
                        raise RuntimeError(f"Execution error: {error_msg}")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Generation timed out after {timeout}s")

    async def _parse_history_result(
        self,
        prompt_id: str,
        history: dict[str, Any],
    ) -> GenerationResult:
        """Parse history data into GenerationResult."""
        images: list[GeneratedImage] = []
        outputs = history.get("outputs", {})

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    images.append(
                        GeneratedImage(
                            filename=img_info["filename"],
                            subfolder=img_info.get("subfolder", ""),
                            type=img_info.get("type", "output"),
                        )
                    )

        return GenerationResult(
            prompt_id=prompt_id,
            images=images,
        )

    async def generate_and_fetch(
        self,
        workflow: dict[str, Any],
        output_format: str = "path",
        use_websocket: bool = True,
    ) -> GenerationResult:
        """Queue workflow, wait for completion, and fetch images."""
        prompt_id = await self.queue_prompt(workflow)

        if use_websocket:
            result: GenerationResult | None = None
            async for update in self.wait_for_completion_websocket(prompt_id):
                if isinstance(update, GenerationResult):
                    result = update
                    break
            if result is None:
                raise RuntimeError("Generation completed but no result received")
        else:
            result = await self.wait_for_completion_polling(prompt_id)

        # Fetch image data if requested
        if output_format == "base64":
            for image in result.images:
                img_bytes = await self.get_image(
                    image.filename,
                    image.subfolder,
                    image.type,
                )
                image.data = base64.b64encode(img_bytes).decode("utf-8")

        return result
