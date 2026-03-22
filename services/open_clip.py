from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config


class OpenClipEmbeddingService:
    def __init__(
        self,
        model_name: str = config.OPEN_CLIP_MODEL_NAME,
        pretrained: str = config.OPEN_CLIP_PRETRAINED,
        cache_dir: str = config.OPEN_CLIP_CACHE_DIR,
        device: str = config.OPEN_CLIP_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.cache_dir = Path(cache_dir)
        self.device = device

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.torch = None

    def embed_image(self, image: np.ndarray | Image.Image) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: list[np.ndarray | Image.Image]) -> list[list[float]]:
        if not images:
            return []

        self._load_model()
        assert self.torch is not None
        assert self.preprocess is not None
        assert self.model is not None

        pil_images = [self._to_pil_image(image) for image in images]
        image_tensor = self.torch.stack([self.preprocess(image) for image in pil_images]).to(self.device)

        with self.torch.no_grad():
            with self._autocast_context():
                image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._load_model()
        assert self.torch is not None
        assert self.tokenizer is not None
        assert self.model is not None

        text_tokens = self.tokenizer(texts).to(self.device)

        with self.torch.no_grad():
            with self._autocast_context():
                text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().tolist()

    def _load_model(self) -> None:
        if self.model is not None:
            return

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip is required. Install it with `pip install open-clip-torch`."
            ) from exc

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for OpenClipEmbeddingService. Install torch first."
            ) from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            cache_dir=str(self.cache_dir),
        )
        model.eval()
        model = model.to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.torch = torch

    def _to_pil_image(self, image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected image array with shape (H, W, 3).")
            rgb_image = image[:, :, ::-1]
            return Image.fromarray(rgb_image.astype(np.uint8)).convert("RGB")

        raise TypeError("Expected a PIL image or numpy image array.")

    def _autocast_context(self) -> Any:
        assert self.torch is not None
        if self.device.startswith("cuda"):
            return self.torch.autocast("cuda")
        return nullcontext()
