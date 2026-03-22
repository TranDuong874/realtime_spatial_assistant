from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence
import sys

import numpy as np
from PIL import Image
import yaml

import config


class SlowFastEmbeddingService:
    def __init__(
        self,
        checkpoint_path: str | Path = config.SLOWFAST_CHECKPOINT_PATH,
        repo_path: str | Path = config.SLOWFAST_REPO_PATH,
        device: str = config.SLOWFAST_DEVICE,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.repo_path = self._resolve_path(repo_path)
        self.device = device

        self.model = None
        self.torch = None
        self.cfg = None

    def embed_frame(self, frame: np.ndarray | Image.Image) -> list[float]:
        return self.embed_frames([frame])[0]

    def embed_frames(self, frames: list[np.ndarray | Image.Image]) -> list[list[float]]:
        if not frames:
            return []

        clips = [[frame] for frame in frames]
        return self.embed_clips(clips)

    def embed_clip(self, frames: Sequence[np.ndarray | Image.Image]) -> list[float]:
        return self.embed_clips([list(frames)])[0]

    def embed_clips(
        self,
        clips: Sequence[Sequence[np.ndarray | Image.Image]],
    ) -> list[list[float]]:
        if not clips:
            return []

        self._load_model()
        assert self.model is not None
        assert self.torch is not None
        assert self.cfg is not None

        clip_tensor = self._build_batch_tensor(clips).to(self.device)

        with self.torch.no_grad():
            with self._autocast_context():
                embeddings = self._extract_embeddings(clip_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return embeddings.cpu().tolist()

    def _build_batch_tensor(
        self,
        clips: Sequence[Sequence[np.ndarray | Image.Image]],
    ) -> Any:
        assert self.torch is not None

        processed_clips = [self._prepare_clip(clip) for clip in clips]
        return self.torch.stack(processed_clips, dim=0)

    def _prepare_clip(self, frames: Sequence[np.ndarray | Image.Image]) -> Any:
        assert self.torch is not None
        assert self.cfg is not None

        if not frames:
            raise ValueError("Expected at least one frame for SlowFast embedding.")

        clip_frames = self._resample_frames(frames, self.cfg.DATA.NUM_FRAMES)
        tensor_frames = [self._to_tensor(frame) for frame in clip_frames]
        clip = self.torch.stack(tensor_frames, dim=0)
        clip = self._resize_short_side(clip, self.cfg.DATA.TEST_CROP_SIZE)
        clip = self._center_crop(clip, self.cfg.DATA.TEST_CROP_SIZE)
        clip = clip.permute(1, 0, 2, 3)

        mean = self.torch.tensor(self.cfg.DATA.MEAN, dtype=clip.dtype).view(3, 1, 1, 1)
        std = self.torch.tensor(self.cfg.DATA.STD, dtype=clip.dtype).view(3, 1, 1, 1)
        return (clip - mean) / std

    def _extract_embeddings(self, clip_tensor: Any) -> Any:
        assert self.model is not None

        pathways = self._pack_pathways(clip_tensor)
        model = self.model

        x = model.s1(pathways)
        x = model.s1_fuse(x)
        x = model.s2(x)
        x = model.s2_fuse(x)
        for pathway in range(model.num_pathways):
            pool = getattr(model, f"pathway{pathway}_pool")
            x[pathway] = pool(x[pathway])
        x = model.s3(x)
        x = model.s3_fuse(x)
        x = model.s4(x)
        x = model.s4_fuse(x)
        x = model.s5(x)

        pooled = []
        for pathway in range(model.head.num_pathways):
            avg_pool = getattr(model.head, f"pathway{pathway}_avgpool")
            pooled.append(avg_pool(x[pathway]))

        x = self.torch.cat(pooled, dim=1)
        x = x.permute(0, 2, 3, 4, 1)
        if hasattr(model.head, "dropout"):
            x = model.head.dropout(x)

        return x.mean(dim=(1, 2, 3))

    def _pack_pathways(self, clip_tensor: Any) -> list[Any]:
        assert self.torch is not None
        assert self.cfg is not None

        fast_pathway = clip_tensor
        slow_indices = self.torch.linspace(
            0,
            clip_tensor.shape[2] - 1,
            clip_tensor.shape[2] // self.cfg.SLOWFAST.ALPHA,
        ).long()
        slow_pathway = self.torch.index_select(clip_tensor, 2, slow_indices.to(clip_tensor.device))
        return [slow_pathway, fast_pathway]

    def _load_model(self) -> None:
        if self.model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for SlowFastEmbeddingService. Install torch first."
            ) from exc

        self.torch = torch

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SlowFast checkpoint not found: {self.checkpoint_path}. "
                "Put `SlowFast.pyth` in the project's models/ folder."
            )

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"SlowFast repo not found: {self.repo_path}. "
                "Vendor `epic-kitchens-slowfast` under thirdparty/ first."
            )

        self._install_vendor_shims()

        repo_path_str = str(self.repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        from slowfast.models.video_model_builder import SlowFast

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        cfg_text = checkpoint.get("cfg")
        if not isinstance(cfg_text, str):
            raise ValueError("SlowFast checkpoint does not contain a YAML cfg string.")

        cfg_dict = yaml.safe_load(cfg_text)
        cfg_dict["DETECTION"]["ENABLE"] = False
        cfg = self._to_namespace(cfg_dict)

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        model = SlowFast(cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model = model.to(device)

        self.model = model
        self.cfg = cfg

    def _to_tensor(self, frame: np.ndarray | Image.Image) -> Any:
        assert self.torch is not None

        rgb_frame = self._to_rgb_array(frame)
        if rgb_frame.dtype.kind in {"f"} and rgb_frame.max() <= 1.0:
            frame_data = rgb_frame.astype(np.float32)
        else:
            frame_data = rgb_frame.astype(np.float32) / 255.0

        return self.torch.from_numpy(frame_data.copy()).permute(2, 0, 1)

    def _to_rgb_array(self, frame: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"))

        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Expected frame array with shape (H, W, 3).")
            return frame[:, :, ::-1]

        raise TypeError("Expected a PIL image or numpy image array.")

    def _resample_frames(
        self,
        frames: Sequence[np.ndarray | Image.Image],
        target_frames: int,
    ) -> list[np.ndarray | Image.Image]:
        if len(frames) == target_frames:
            return list(frames)

        sample_points = np.linspace(0, len(frames) - 1, num=target_frames)
        indices = np.rint(sample_points).astype(int)
        return [frames[index] for index in indices]

    def _resize_short_side(self, clip: Any, size: int) -> Any:
        assert self.torch is not None

        _, _, height, width = clip.shape
        if min(height, width) == size:
            return clip

        if height < width:
            new_height = size
            new_width = int(round(width * size / height))
        else:
            new_width = size
            new_height = int(round(height * size / width))

        return self.torch.nn.functional.interpolate(
            clip,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

    def _center_crop(self, clip: Any, size: int) -> Any:
        _, _, height, width = clip.shape
        y_offset = max((height - size) // 2, 0)
        x_offset = max((width - size) // 2, 0)
        return clip[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    def _install_vendor_shims(self) -> None:
        if "fvcore.common.registry" not in sys.modules:
            registry_module = ModuleType("fvcore.common.registry")

            class Registry:
                def __init__(self, name: str) -> None:
                    self._name = name
                    self._objects: dict[str, Any] = {}

                def register(self) -> Any:
                    def decorator(obj: Any) -> Any:
                        self._objects[obj.__name__] = obj
                        return obj

                    return decorator

                def get(self, name: str) -> Any:
                    return self._objects[name]

            registry_module.Registry = Registry
            sys.modules["fvcore.common.registry"] = registry_module

        if "fvcore.common" not in sys.modules:
            common_module = ModuleType("fvcore.common")
            common_module.registry = sys.modules["fvcore.common.registry"]
            sys.modules["fvcore.common"] = common_module

        if "fvcore.nn.weight_init" not in sys.modules:
            assert self.torch is not None
            weight_init_module = ModuleType("fvcore.nn.weight_init")

            def c2_msra_fill(module: Any) -> None:
                self.torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    self.torch.nn.init.constant_(module.bias, 0)

            weight_init_module.c2_msra_fill = c2_msra_fill
            sys.modules["fvcore.nn.weight_init"] = weight_init_module

        if "fvcore.nn" not in sys.modules:
            nn_module = ModuleType("fvcore.nn")
            nn_module.weight_init = sys.modules["fvcore.nn.weight_init"]
            sys.modules["fvcore.nn"] = nn_module

        if "fvcore" not in sys.modules:
            fvcore_module = ModuleType("fvcore")
            fvcore_module.common = sys.modules["fvcore.common"]
            fvcore_module.nn = sys.modules["fvcore.nn"]
            sys.modules["fvcore"] = fvcore_module

        if "detectron2.layers" not in sys.modules:
            detectron_layers = ModuleType("detectron2.layers")

            class ROIAlign:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise NotImplementedError(
                        "ROIAlign is unavailable in this lightweight SlowFast inference setup."
                    )

            detectron_layers.ROIAlign = ROIAlign
            sys.modules["detectron2.layers"] = detectron_layers

        if "detectron2" not in sys.modules:
            detectron2_module = ModuleType("detectron2")
            detectron2_module.layers = sys.modules["detectron2.layers"]
            sys.modules["detectron2"] = detectron2_module

    def _to_namespace(self, value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: self._to_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [self._to_namespace(item) for item in value]
        return value

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _autocast_context(self) -> Any:
        assert self.torch is not None
        if self.device.startswith("cuda"):
            return self.torch.autocast("cuda")
        return nullcontext()
