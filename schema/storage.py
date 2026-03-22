from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class FrameRecord:
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    frame_path: str
    ocr_text: str | None = None
    yolo_json: list[dict[str, Any]] | dict[str, Any] | None = None
    slam_json: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ClipRecord:
    clip_id: str
    start_frame_id: str
    end_frame_id: str
    start_s: float
    end_s: float
    feature_path: str


@dataclass(slots=True, frozen=True)
class SegmentRecord:
    segment_id: str
    start_frame_id: str
    end_frame_id: str
    start_s: float
    end_s: float
    action_text: str
    score: float
    verb_label: str | None = None
    noun_label: str | None = None


@dataclass(slots=True, frozen=True)
class SegmentFrameRecord:
    segment_id: str
    frame_id: str
    role: str


@dataclass(slots=True, frozen=True)
class SegmentClipRecord:
    segment_id: str
    clip_id: str


@dataclass(slots=True, frozen=True)
class StoredSegment:
    segment: SegmentRecord
    frames: list[SegmentFrameRecord] = field(default_factory=list)
    clips: list[SegmentClipRecord] = field(default_factory=list)


__all__ = [
    "ClipRecord",
    "FrameRecord",
    "SegmentClipRecord",
    "SegmentFrameRecord",
    "SegmentRecord",
    "StoredSegment",
]
