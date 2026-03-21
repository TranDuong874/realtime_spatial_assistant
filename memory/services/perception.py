from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import config


class YoloDetectionService:
    def __init__(
        self,
        model_path: str | Path = config.YOLO_MODEL_PATH,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.YOLO_IOU_THRESHOLD,
        device: str | None = config.YOLO_DEVICE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def detect_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return self.detect_frames([frame])[0]

    def detect_frames(self, frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        if not frames:
            return []

        model = self._get_model()
        results = model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return [self._format_result(result) for result in results]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for YoloDetectionService. "
                    "Install it with `pip install ultralytics`."
                ) from exc

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"YOLO checkpoint not found: {self.model_path}. "
                    "Put the checkpoint in the project's models/ folder."
                )

            self._model = YOLO(str(self.model_path))
        return self._model

    def _format_result(self, result: Any) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None:
            return []

        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
        name_map = result.names
        detections: list[dict[str, Any]] = []

        for xyxy, confidence, class_id in zip(xyxy_values, confidence_values, class_values):
            x1, y1, x2, y2 = xyxy
            class_index = int(class_id)
            detections.append(
                {
                    "class_id": class_index,
                    "class_name": name_map.get(class_index, str(class_index)),
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_xywh": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
            )

        return detections
