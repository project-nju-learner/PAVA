from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class OpenClipBackend:
    """Lazy optional OpenCLIP encoder; importing this module has no model side effects."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if model_name.strip().lower().startswith("hf-hub:"):
            raise ValueError(
                "hf-hub model names are not allowed by the offline-only backend"
            )
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self.device: str | None = None
        self._torch = None
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP is optional; install torch, open_clip_torch, and Pillow "
                "in the caller-owned environment before using this backend"
            ) from exc
        if not self.checkpoint_path.is_file():
            raise RuntimeError(
                f"local OpenCLIP checkpoint does not exist: {self.checkpoint_path}"
            )
        model_config = open_clip.get_model_config(self.model_name)
        self._validate_offline_model_config(model_config)
        device = self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=str(self.checkpoint_path),
            device=device,
        )
        model.eval()
        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self.device = device

    @staticmethod
    def _validate_offline_model_config(model_config: object) -> None:
        if not isinstance(model_config, dict):
            raise RuntimeError("model_name must resolve to a bundled OpenCLIP config")
        text_config = model_config.get("text_cfg", {})
        if not isinstance(text_config, dict):
            raise RuntimeError("OpenCLIP text_cfg must be a local bundled config")
        if text_config.get("hf_model_name") or text_config.get("hf_tokenizer_name"):
            raise RuntimeError(
                "OpenCLIP configs with external Hugging Face text components are "
                "not allowed by the offline-only backend"
            )

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if not images_rgb:
            raise ValueError("at least one RGB image is required")
        self._load()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to encode RGB images") from exc
        tensors = []
        for image_rgb in images_rgb:
            array = np.asarray(image_rgb)
            if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
                raise ValueError("each image must have shape (height, width, at least 3)")
            if array.dtype != np.uint8:
                raise ValueError("each image must use uint8 RGB values")
            array = array[:, :, :3]
            tensors.append(self._preprocess(Image.fromarray(array, mode="RGB")))
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype(np.float32, copy=False)

    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items or any(not item.strip() for item in items):
            raise ValueError("texts must contain at least one non-empty string")
        self._load()
        tokens = self._tokenizer(items).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype(np.float32, copy=False)
