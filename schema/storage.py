from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class FrameRecord:
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    frame_path: str
    ocr_text: str | None = None
    ocr_json: list[dict[str, Any]] | dict[str, Any] | None = None
    yolo_json: list[dict[str, Any]] | dict[str, Any] | None = None
    slam_json: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class SegmentRecord:
    segment_id: str
    start_frame_id: str
    end_frame_id: str
    start_frame_idx: int
    end_frame_idx: int
    start_s: float
    end_s: float
    action_text: str
    score: float
    verb_label: str | None = None
    noun_label: str | None = None
    rep_frame_start_id: str | None = None
    rep_frame_mid_id: str | None = None
    rep_frame_end_id: str | None = None


@dataclass(slots=True, frozen=True)
class PooledWindowRecord:
    window_id: str
    video_id: str
    start_frame_idx: int
    end_frame_idx: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    frame_count: int


__all__ = [
    "FrameRecord",
    "PooledWindowRecord",
    "SegmentRecord",
]
