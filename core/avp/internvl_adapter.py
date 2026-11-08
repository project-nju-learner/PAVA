"""InternVL transport for AVP's existing OpenAI-compatible provider path."""

from __future__ import annotations

from typing import Any, Optional

from .qwen_adapter import OpenAICompatibleQwen, QwenClient


class OpenAICompatibleInternVL(OpenAICompatibleQwen):
    """Translate only processor-specific request fields for InternVL."""

    @staticmethod
    def video_num_frames() -> int:
        # InternVL3.5 is served with a 32K context. Keep AVP's requested fps,
        # but cap long clips before they overflow that fixed model context.
        return 96

    @staticmethod
    def processor_kwargs_for(
        *, fps: Optional[float], max_pixels: Optional[int], has_video: bool
    ) -> dict[str, Any]:
        del fps
        if has_video:
            return {}
        if max_pixels is None:
            return {}
        patch_pixels = 448 * 448
        max_dynamic_patch = max(
            1, min(12, (max_pixels + patch_pixels - 1) // patch_pixels)
        )
        return {"max_dynamic_patch": max_dynamic_patch}

class InternVLClient(QwenClient):
    """QwenClient-compatible AVP client with InternVL processor fields."""

    def initialize_client(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleInternVL(self.base_url, debug=self.debug)
            if self.debug:
                print(f"Initialized InternVL OpenAI-compatible client: {self.base_url}")
