from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class FrameInput:
    id: str
    timestamp_ms: int
    frame_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FrameMemory:
    frame: FrameInput
    embedding: list[float]
    yolo_detections: list[dict[str, Any]] = field(default_factory=list)


__all__ = [
    "FrameInput",
    "FrameMemory",
]
