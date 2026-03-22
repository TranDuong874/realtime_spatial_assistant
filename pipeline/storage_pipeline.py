from __future__ import annotations

from qdrant_client.models import Filter

from database.postgres.client import PostgresClient
from database.qdrant.client import QdrantClientWrapper
from schema import FrameRecord, PooledWindowRecord, SegmentRecord


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
        *,
        video_id: str | None = None,
    ) -> None:
        if self.postgres is None:
            raise RuntimeError("Call initialize() before store_frame().")

        self.postgres.upsert_frame(
            frame_id=frame.frame_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            frame_path=frame.frame_path,
            ocr_text=frame.ocr_text,
            ocr_json=frame.ocr_json,
            yolo_json=frame.yolo_json,
            slam_json=frame.slam_json,
        )
        if openclip_embedding is not None:
            if self.qdrant is None:
                raise RuntimeError("Call initialize() before storing frame embeddings.")
            payload = {
                "frame_idx": frame.frame_idx,
                "timestamp_ms": frame.timestamp_ms,
            }
            if video_id is not None:
                payload["video_id"] = video_id
            self.qdrant.upsert_frame_point(frame.frame_id, openclip_embedding, payload=payload)

    def store_segment(
        self,
        segment: SegmentRecord,
        semantic_embedding: list[float] | None = None,
    ) -> None:
        if self.postgres is None:
            raise RuntimeError("Call initialize() before store_segment().")

        self.postgres.upsert_segment(
            segment_id=segment.segment_id,
            start_frame_id=segment.start_frame_id,
            end_frame_id=segment.end_frame_id,
            start_frame_idx=segment.start_frame_idx,
            end_frame_idx=segment.end_frame_idx,
            start_s=segment.start_s,
            end_s=segment.end_s,
            verb_label=segment.verb_label,
            noun_label=segment.noun_label,
            action_text=segment.action_text,
            score=segment.score,
            rep_frame_start_id=segment.rep_frame_start_id,
            rep_frame_mid_id=segment.rep_frame_mid_id,
            rep_frame_end_id=segment.rep_frame_end_id,
        )
        if semantic_embedding is not None:
            if self.qdrant is None:
                raise RuntimeError("Call initialize() before storing segment embeddings.")
            self.qdrant.upsert_segment_point(
                segment.segment_id,
                semantic_embedding,
                payload={
                    "action_text": segment.action_text,
                    "verb_label": segment.verb_label,
                    "noun_label": segment.noun_label,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "score": segment.score,
                    "start_frame_id": segment.start_frame_id,
                    "end_frame_id": segment.end_frame_id,
                    "start_frame_idx": segment.start_frame_idx,
                    "end_frame_idx": segment.end_frame_idx,
                    "rep_frame_start_id": segment.rep_frame_start_id,
                    "rep_frame_mid_id": segment.rep_frame_mid_id,
                    "rep_frame_end_id": segment.rep_frame_end_id,
                },
            )

    def store_window(
        self,
        window: PooledWindowRecord,
        semantic_embedding: list[float],
    ) -> None:
        if self.qdrant is None:
            raise RuntimeError("Call initialize() before store_window().")

        self.qdrant.upsert_window_point(
            window.window_id,
            semantic_embedding,
            payload={
                "video_id": window.video_id,
                "start_frame_idx": window.start_frame_idx,
                "end_frame_idx": window.end_frame_idx,
                "start_timestamp_ms": window.start_timestamp_ms,
                "end_timestamp_ms": window.end_timestamp_ms,
                "frame_count": window.frame_count,
            },
        )

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
            "ocr_json": frame.ocr_json,
            "yolo_json": frame.yolo_json,
            "slam_json": frame.slam_json,
        }

    def hydrate_segment(self, segment_id: str) -> dict | None:
        if self.postgres is None:
            raise RuntimeError("Postgres client is not initialized.")

        segment = self.postgres.get_segment(segment_id)
        if segment is None:
            return None

        return {
            "segment_id": segment.segment_id,
            "start_frame_id": segment.start_frame_id,
            "end_frame_id": segment.end_frame_id,
            "start_frame_idx": segment.start_frame_idx,
            "end_frame_idx": segment.end_frame_idx,
            "start_s": segment.start_s,
            "end_s": segment.end_s,
            "verb_label": segment.verb_label,
            "noun_label": segment.noun_label,
            "action_text": segment.action_text,
            "score": segment.score,
            "rep_frame_start_id": segment.rep_frame_start_id,
            "rep_frame_mid_id": segment.rep_frame_mid_id,
            "rep_frame_end_id": segment.rep_frame_end_id,
        }

    def search_frames(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        return self.search_frames_filtered(query_embedding, limit=limit, query_filter=None)

    def search_frames_filtered(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search_frames(query_embedding, limit=limit, query_filter=query_filter)
        results: list[dict] = []
        for hit in hits:
            frame_id = hit.payload.get("frame_id", str(hit.id))
            record = self.hydrate_frame(frame_id)
            if record is None:
                continue
            record["score"] = float(hit.score)
            record["payload"] = dict(hit.payload)
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
            record["payload"] = dict(hit.payload)
            results.append(record)
        return results

    def search_windows(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search_windows(query_embedding, limit=limit)
        results: list[dict] = []
        for hit in hits:
            payload = dict(hit.payload)
            payload["score"] = float(hit.score)
            results.append(payload)
        return results

    def search_frames_in_window(
        self,
        query_embedding: list[float],
        *,
        video_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        limit: int = 10,
    ) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        frame_filter = self.qdrant.build_frame_scope_filter(
            video_id=video_id,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
        return self.search_frames_filtered(query_embedding, limit=limit, query_filter=frame_filter)

    def hierarchical_frame_search(
        self,
        query_embedding: list[float],
        *,
        window_limit: int = 5,
        frame_limit_per_window: int = 5,
    ) -> dict:
        windows = self.search_windows(query_embedding, limit=window_limit)
        grouped_frames: list[dict] = []
        flattened_frames: list[dict] = []
        seen_frame_ids: set[str] = set()
        for window in windows:
            frames = self.search_frames_in_window(
                query_embedding,
                video_id=str(window["video_id"]),
                start_frame_idx=int(window["start_frame_idx"]),
                end_frame_idx=int(window["end_frame_idx"]),
                limit=frame_limit_per_window,
            )
            grouped_frames.append(
                {
                    "window": window,
                    "frames": frames,
                }
            )
            for frame in frames:
                frame_id = str(frame["frame_id"])
                if frame_id in seen_frame_ids:
                    continue
                seen_frame_ids.add(frame_id)
                flattened_frames.append(
                    {
                        **frame,
                        "window_id": window["window_id"],
                        "window_start_frame_idx": window["start_frame_idx"],
                        "window_end_frame_idx": window["end_frame_idx"],
                    }
                )
        flattened_frames.sort(key=lambda item: float(item["score"]), reverse=True)
        return {
            "windows": windows,
            "window_frames": grouped_frames,
            "frames": flattened_frames,
        }


__all__ = ["EvaluationStoragePipeline"]
