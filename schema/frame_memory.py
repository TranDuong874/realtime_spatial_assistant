from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class FrameInput:
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    frame_path: str
    ocr_text: str | None = None
    ocr_json: list[dict[str, Any]] | dict[str, Any] | None = None
    yolo_json: list[dict[str, Any]] | dict[str, Any] | None = None
    slam_json: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return self.frame_id

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0

    @property
    def metadata(self) -> dict[str, Any]:
        return {}


@dataclass(slots=True, frozen=True)
class FrameMemory:
    frame: FrameInput
    embedding: list[float]
    yolo_detections: list[dict[str, Any]] = field(default_factory=list)


__all__ = [
    "FrameInput",
    "FrameMemory",
]
