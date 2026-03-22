from __future__ import annotations

from database.postgres.client import PostgresClient
from database.qdrant.client import QdrantClientWrapper
from schema import ClipRecord, FrameRecord, SegmentClipRecord, SegmentFrameRecord, SegmentRecord


class EvaluationStoragePipeline:
    def __init__(
        self,
        postgres: PostgresClient | None = None,
        qdrant: QdrantClientWrapper | None = None,
    ) -> None:
        self.postgres = postgres
        self.qdrant = qdrant

    def initialize(self, recreate_qdrant_collections: bool = False) -> None:
        if self.postgres is None:
            self.postgres = PostgresClient()
        if self.qdrant is None:
            self.qdrant = QdrantClientWrapper()
        self.postgres.init_db()
        self.qdrant.init_collections(recreate=recreate_qdrant_collections)

    def reset(self) -> None:
        if self.postgres is None:
            self.postgres = PostgresClient()
        if self.qdrant is None:
            self.qdrant = QdrantClientWrapper()
        self.postgres.reset_db()
        self.qdrant.reset_collections()

    def store_frame(
        self,
        frame: FrameRecord,
        openclip_embedding: list[float] | None = None,
    ) -> None:
        if self.postgres is None:
            raise RuntimeError("Call initialize() before store_frame().")

        self.postgres.upsert_frame(
            frame_id=frame.frame_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            frame_path=frame.frame_path,
            ocr_text=frame.ocr_text,
            yolo_json=frame.yolo_json,
            slam_json=frame.slam_json,
        )
        if openclip_embedding is not None:
            if self.qdrant is None:
                raise RuntimeError("Call initialize() before storing frame embeddings.")
            self.qdrant.upsert_frame_point(frame.frame_id, openclip_embedding)

    def store_clip(self, clip: ClipRecord) -> None:
        if self.postgres is None:
            raise RuntimeError("Call initialize() before store_clip().")

        self.postgres.upsert_clip(
            clip_id=clip.clip_id,
            start_frame_id=clip.start_frame_id,
            end_frame_id=clip.end_frame_id,
            start_s=clip.start_s,
            end_s=clip.end_s,
            feature_path=clip.feature_path,
        )

    def store_segment(
        self,
        segment: SegmentRecord,
        representative_frames: list[SegmentFrameRecord] | None = None,
        covered_clips: list[SegmentClipRecord] | None = None,
        semantic_embedding: list[float] | None = None,
    ) -> None:
        if self.postgres is None:
            raise RuntimeError("Call initialize() before store_segment().")

        self.postgres.upsert_segment(
            segment_id=segment.segment_id,
            start_frame_id=segment.start_frame_id,
            end_frame_id=segment.end_frame_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            verb_label=segment.verb_label,
            noun_label=segment.noun_label,
            action_text=segment.action_text,
            score=segment.score,
        )
        self.postgres.replace_segment_frames(
            segment.segment_id,
            [] if representative_frames is None else [
                {"frame_id": item.frame_id, "role": item.role}
                for item in representative_frames
            ],
        )
        self.postgres.replace_segment_clips(
            segment.segment_id,
            [] if covered_clips is None else [item.clip_id for item in covered_clips],
        )
        if semantic_embedding is not None:
            if self.qdrant is None:
                raise RuntimeError("Call initialize() before storing segment embeddings.")
            self.qdrant.upsert_segment_point(segment.segment_id, semantic_embedding)

    def hydrate_frame(self, frame_id: str) -> dict | None:
        if self.postgres is None:
            raise RuntimeError("Postgres client is not initialized.")

        frame = self.postgres.get_frame(frame_id)
        if frame is None:
            return None

        return {
            "frame_id": frame.frame_id,
            "frame_idx": frame.frame_idx,
            "timestamp_ms": frame.timestamp_ms,
            "timestamp_s": frame.timestamp_ms / 1000.0,
            "frame_path": frame.frame_path,
            "ocr_text": frame.ocr_text,
            "yolo_json": frame.yolo_json,
            "slam_json": frame.slam_json,
        }

    def hydrate_segment(self, segment_id: str) -> dict | None:
        if self.postgres is None:
            raise RuntimeError("Postgres client is not initialized.")

        segment = self.postgres.get_segment(segment_id)
        if segment is None:
            return None

        representative_frames = self.postgres.get_segment_frames(segment_id)
        covered_clips = self.postgres.get_segment_clips(segment_id)
        return {
            "segment_id": segment.segment_id,
            "start_frame_id": segment.start_frame_id,
            "end_frame_id": segment.end_frame_id,
            "start_s": segment.start_s,
            "end_s": segment.end_s,
            "verb_label": segment.verb_label,
            "noun_label": segment.noun_label,
            "action_text": segment.action_text,
            "score": segment.score,
            "segment_frames": [
                {
                    "frame_id": item.frame_id,
                    "role": item.role,
                }
                for item in representative_frames
            ],
            "segment_clips": [
                {
                    "clip_id": item.clip_id,
                }
                for item in covered_clips
            ],
        }

    def search_frames(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search_frames(query_embedding, limit=limit)
        results: list[dict] = []
        for hit in hits:
            frame_id = hit.payload.get("frame_id", str(hit.id))
            record = self.hydrate_frame(frame_id)
            if record is None:
                continue
            record["score"] = float(hit.score)
            results.append(record)
        return results

    def search_segments(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search_segments(query_embedding, limit=limit)
        results: list[dict] = []
        for hit in hits:
            segment_id = hit.payload.get("segment_id", str(hit.id))
            record = self.hydrate_segment(segment_id)
            if record is None:
                continue
            record["score"] = float(hit.score)
            results.append(record)
        return results


__all__ = ["EvaluationStoragePipeline"]
