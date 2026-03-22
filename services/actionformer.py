from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np

import config
from schema import ActionSegmentPrediction, ActionWindowInput, ActionWindowPrediction


@dataclass(slots=True)
class _ActionFormerModelBundle:
    kind: str
    model: Any
    cfg: dict[str, Any]
    checkpoint_path: Path


class ActionFormerSegmentService:
    def __init__(
        self,
        repo_path: str | Path = config.ACTIONFORMER_REPO_PATH,
        verb_config_path: str | Path = config.ACTIONFORMER_EPIC_VERB_CONFIG_PATH,
        noun_config_path: str | Path = config.ACTIONFORMER_EPIC_NOUN_CONFIG_PATH,
        verb_checkpoint_path: str | Path = config.ACTIONFORMER_EPIC_VERB_CHECKPOINT_PATH,
        noun_checkpoint_path: str | Path = config.ACTIONFORMER_EPIC_NOUN_CHECKPOINT_PATH,
        device: str = config.ACTIONFORMER_DEVICE,
        default_fps: float = config.ACTIONFORMER_DEFAULT_FPS,
        feat_stride: int = config.ACTIONFORMER_FEAT_STRIDE,
        feat_num_frames: int = config.ACTIONFORMER_FEAT_NUM_FRAMES,
        input_dim: int = config.ACTIONFORMER_INPUT_DIM,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.repo_path = self._resolve_path(repo_path)
        self.verb_config_path = self._resolve_path(verb_config_path)
        self.noun_config_path = self._resolve_path(noun_config_path)
        self.verb_checkpoint_path = self._resolve_path(verb_checkpoint_path)
        self.noun_checkpoint_path = self._resolve_path(noun_checkpoint_path)
        self.device = device
        self.default_fps = default_fps
        self.feat_stride = feat_stride
        self.feat_num_frames = feat_num_frames
        self.input_dim = input_dim

        self.torch = None
        self._verb_bundle: _ActionFormerModelBundle | None = None
        self._noun_bundle: _ActionFormerModelBundle | None = None

    def infer_verb_segments(self, window: ActionWindowInput) -> list[ActionSegmentPrediction]:
        bundle = self._get_verb_bundle()
        return self._infer_segments(bundle, window)

    def infer_noun_segments(self, window: ActionWindowInput) -> list[ActionSegmentPrediction]:
        bundle = self._get_noun_bundle()
        return self._infer_segments(bundle, window)

    def infer_window(self, window: ActionWindowInput) -> ActionWindowPrediction:
        return ActionWindowPrediction(
            verb_segments=self.infer_verb_segments(window),
            noun_segments=self.infer_noun_segments(window),
        )

    def infer_segments(self, window: ActionWindowInput) -> ActionWindowPrediction:
        return self.infer_window(window)

    def _infer_segments(
        self,
        bundle: _ActionFormerModelBundle,
        window: ActionWindowInput,
    ) -> list[ActionSegmentPrediction]:
        assert self.torch is not None

        resolved_fps = window.fps
        resolved_stride = window.feat_stride
        resolved_num_frames = window.feat_num_frames
        feature_tensor = self._to_feature_tensor(window.slowfast_features)
        resolved_duration = (
            self._infer_duration_seconds(feature_tensor.shape[-1], resolved_fps, resolved_stride, resolved_num_frames)
            if window.duration_seconds is None
            else window.duration_seconds
        )

        video_item = {
            "video_id": window.video_id,
            "feats": feature_tensor,
            "segments": None,
            "labels": None,
            "fps": resolved_fps,
            "duration": resolved_duration,
            "feat_stride": resolved_stride,
            "feat_num_frames": resolved_num_frames,
        }

        with self.torch.no_grad():
            results = bundle.model([video_item])

        if not results:
            return []

        result = results[0]
        segments = result["segments"].tolist()
        scores = result["scores"].tolist()
        labels = result["labels"].tolist()

        formatted: list[ActionSegmentPrediction] = []
        for segment, score, label in zip(segments, scores, labels):
            start_seconds = float(segment[0]) + window.window_start_seconds
            end_seconds = float(segment[1]) + window.window_start_seconds
            formatted.append(
                ActionSegmentPrediction(
                    kind=bundle.kind,
                    video_id=window.video_id,
                    label_id=int(label),
                    score=float(score),
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    duration_seconds=max(0.0, end_seconds - start_seconds),
                    window_start_seconds=window.window_start_seconds,
                )
            )

        return formatted

    def _get_verb_bundle(self) -> _ActionFormerModelBundle:
        if self._verb_bundle is None:
            self._verb_bundle = self._load_bundle(
                kind="verb",
                config_path=self.verb_config_path,
                checkpoint_path=self.verb_checkpoint_path,
            )
        return self._verb_bundle

    def _get_noun_bundle(self) -> _ActionFormerModelBundle:
        if self._noun_bundle is None:
            self._noun_bundle = self._load_bundle(
                kind="noun",
                config_path=self.noun_config_path,
                checkpoint_path=self.noun_checkpoint_path,
            )
        return self._noun_bundle

    def _load_bundle(
        self,
        *,
        kind: str,
        config_path: Path,
        checkpoint_path: Path,
    ) -> _ActionFormerModelBundle:
        self._ensure_runtime_ready()
        assert self.torch is not None

        if not config_path.exists():
            raise FileNotFoundError(f"ActionFormer config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ActionFormer checkpoint not found: {checkpoint_path}")

        from libs.core import load_config
        from libs.modeling import make_meta_arch

        cfg = load_config(str(config_path))
        device = self.device
        if device.startswith("cuda") and not self.torch.cuda.is_available():
            device = "cpu"
        self.device = device

        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        checkpoint = self.torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
        if state_dict is None:
            raise ValueError(f"ActionFormer checkpoint is missing state_dict data: {checkpoint_path}")
        normalized_state_dict = self._normalize_state_dict_keys(state_dict)
        model.load_state_dict(normalized_state_dict)
        model.eval()
        model = model.to(device)

        return _ActionFormerModelBundle(
            kind=kind,
            model=model,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
        )

    def _ensure_runtime_ready(self) -> None:
        if self.torch is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for ActionFormerSegmentService. Install torch first."
            ) from exc

        self.torch = torch

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"ActionFormer repo not found: {self.repo_path}. "
                "Vendor `actionformer_release` under thirdparty/ first."
            )

        utils_path = self.repo_path / "libs" / "utils"
        for path in (utils_path, self.repo_path):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _to_feature_tensor(self, slowfast_features: Sequence[Sequence[float]] | np.ndarray) -> Any:
        assert self.torch is not None

        array = np.asarray(slowfast_features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Expected SlowFast features with shape (T, D) or (D, T).")

        if array.shape[1] == self.input_dim:
            array = array.T
        elif array.shape[0] == self.input_dim:
            pass
        else:
            raise ValueError(
                f"Expected SlowFast feature dimension {self.input_dim}, got shape {tuple(array.shape)}."
            )

        if array.shape[1] == 0:
            raise ValueError("ActionFormer inference requires at least one SlowFast feature vector.")

        return self.torch.from_numpy(array.copy())

    def _infer_duration_seconds(
        self,
        num_feature_steps: int,
        fps: float,
        feat_stride: int,
        feat_num_frames: int,
    ) -> float:
        if num_feature_steps <= 0:
            return 0.0
        covered_frames = (num_feature_steps - 1) * feat_stride + feat_num_frames
        return covered_frames / fps

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _normalize_state_dict_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                normalized[key.removeprefix("module.")] = value
            else:
                normalized[key] = value
        return normalized
