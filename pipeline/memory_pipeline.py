from __future__ import annotations

from database.postgres.client import PostgresClient
from database.qdrant.client import QdrantClientWrapper
from schema import FrameInput, FrameMemory


class FrameMemoryPipeline:
    def __init__(
        self,
        postgres: PostgresClient | None = None,
        qdrant: QdrantClientWrapper | None = None,
    ) -> None:
        self.postgres = postgres
        self.qdrant = qdrant

    def initialize(self, recreate_qdrant_collection: bool = False) -> None:
        if self.postgres is None:
            self.postgres = PostgresClient()
        if self.qdrant is None:
            self.qdrant = QdrantClientWrapper()
        self.postgres.init_db()
        self.qdrant.init_collections(recreate=recreate_qdrant_collection)

    def create_frame_memory(
        self,
        frame: FrameInput,
        embedding: list[float],
        yolo_detections: list[dict] | None = None,
    ) -> FrameMemory:
        return FrameMemory(
            frame=frame,
            embedding=embedding,
            yolo_detections=yolo_detections or [],
        )

    def store_frame_memory(self, frame_memory: FrameMemory) -> None:
        if self.postgres is None or self.qdrant is None:
            raise RuntimeError("Call initialize() before store_frame_memory().")

        self.postgres.upsert_frame(
            frame_id=frame_memory.frame.frame_id,
            frame_idx=frame_memory.frame.frame_idx,
            timestamp_ms=frame_memory.frame.timestamp_ms,
            frame_path=frame_memory.frame.frame_path,
            ocr_text=frame_memory.frame.ocr_text,
            ocr_json=frame_memory.frame.ocr_json,
            yolo_json=frame_memory.yolo_detections or frame_memory.frame.yolo_json,
            slam_json=frame_memory.frame.slam_json,
        )
        self.qdrant.upsert_frame_point(frame_memory.frame.frame_id, frame_memory.embedding)

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

    def search_frames(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search_frames(query_embedding, limit=limit)
        results: list[dict] = []

        for hit in hits:
            frame_id = hit.payload.get("frame_id", str(hit.id))
            frame_record = self.hydrate_frame(frame_id)
            if frame_record is None:
                continue
            frame_record["score"] = float(hit.score)
            results.append(frame_record)

        return results
