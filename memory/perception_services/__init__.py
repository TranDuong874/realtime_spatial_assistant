from __future__ import annotations

from typing import Any

from memory.schema import FrameInput
from memory.perception_services.ocr import OCRService
from memory.perception_services.segmentation import SegmentationService
from memory.perception_services.yolo import YoloDetectionService


class StubPerceptionPipeline:
    """Collects enrichment outputs without doing actual model inference yet."""

    def __init__(
        self,
        detector: YoloDetectionService | None = None,
        segmenter: SegmentationService | None = None,
        ocr: OCRService | None = None,
    ) -> None:
        self.detector = detector or YoloDetectionService()
        self.segmenter = segmenter or SegmentationService()
        self.ocr = ocr or OCRService()

    def enrich(self, frame: FrameInput) -> dict[str, Any]:
        return {
            "detected_objects": self.detector.run(frame),
            "segments": self.segmenter.run(frame),
            "ocr_text": self.ocr.run(frame),
            "raw_payload": {"frame_id": frame.id},
        }


__all__ = [
    "OCRService",
    "SegmentationService",
    "StubPerceptionPipeline",
    "YoloDetectionService",
]
