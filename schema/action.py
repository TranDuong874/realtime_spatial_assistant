from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True, frozen=True)
class SlowFastClipEmbedding:
    clip_index: int
    start_seconds: float
    end_seconds: float
    embedding: list[float]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(slots=True, frozen=True)
class ActionWindowInput:
    video_id: str
    slowfast_features: Sequence[Sequence[float]]
    window_start_seconds: float = 0.0
    fps: float = 30.0
    feat_stride: int = 16
    feat_num_frames: int = 32
    duration_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class ActionLabel:
    id: int
    key: str
    category: str = ""


@dataclass(slots=True, frozen=True)
class ActionSegmentPrediction:
    kind: str
    video_id: str
    label_id: int
    score: float
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_start_seconds: float


@dataclass(slots=True, frozen=True)
class LabeledActionSegment:
    kind: str
    video_id: str
    label_id: int
    label_name: str
    label_category: str
    score: float
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_start_seconds: float
    merge_count: int = 1


@dataclass(slots=True, frozen=True)
class ActionWindowPrediction:
    verb_segments: list[ActionSegmentPrediction] = field(default_factory=list)
    noun_segments: list[ActionSegmentPrediction] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ActionWindowResult:
    window_index: int
    start_seconds: float
    end_seconds: float
    verb_segments: list[LabeledActionSegment] = field(default_factory=list)
    noun_segments: list[LabeledActionSegment] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ActionSequenceItem:
    start_seconds: float
    end_seconds: float
    phrase: str
    verb: str
    noun: str


@dataclass(slots=True, frozen=True)
class ActionPipelineUpdate:
    window_result: ActionWindowResult
    merged_verb_segments: list[LabeledActionSegment] = field(default_factory=list)
    merged_noun_segments: list[LabeledActionSegment] = field(default_factory=list)
    action_sequence: list[ActionSequenceItem] = field(default_factory=list)


__all__ = [
    "ActionLabel",
    "ActionPipelineUpdate",
    "ActionSegmentPrediction",
    "ActionSequenceItem",
    "ActionWindowInput",
    "ActionWindowPrediction",
    "ActionWindowResult",
    "LabeledActionSegment",
    "SlowFastClipEmbedding",
]
