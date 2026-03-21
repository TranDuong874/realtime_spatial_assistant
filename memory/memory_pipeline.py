from __future__ import annotations

from database.postgres.client import PostgresClient
from database.qdrant.client import QdrantClientWrapper
from memory.schema import FrameInput, FrameMemory


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
        self.qdrant.init_collection(recreate=recreate_qdrant_collection)

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
            frame_id=frame_memory.frame.id,
            timestamp_ms=frame_memory.frame.timestamp_ms,
            frame_path=frame_memory.frame.frame_path,
            metadata=frame_memory.frame.metadata,
        )
        self.postgres.upsert_frame_yolo(
            frame_id=frame_memory.frame.id,
            detections=frame_memory.yolo_detections,
        )
        self.qdrant.upsert_point(
            point_id=frame_memory.frame.id,
            vector=frame_memory.embedding,
            payload={"frame_id": frame_memory.frame.id},
        )

    def hydrate_frame(self, frame_id: str) -> dict | None:
        if self.postgres is None:
            raise RuntimeError("Postgres client is not initialized.")

        frame = self.postgres.get_frame(frame_id)
        if frame is None:
            return None

        yolo = self.postgres.get_frame_yolo(frame_id)
        return {
            "frame_id": frame.id,
            "timestamp_ms": frame.timestamp_ms,
            "frame_path": frame.frame_path,
            "metadata": frame.frame_metadata,
            "yolo_detections": [] if yolo is None else yolo.detections,
        }

    def search_frames(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client is not initialized.")

        hits = self.qdrant.search(query_embedding, limit=limit)
        results: list[dict] = []

        for hit in hits:
            frame_id = hit.payload.get("frame_id", str(hit.id))
            frame_record = self.hydrate_frame(frame_id)
            if frame_record is None:
                continue
            frame_record["score"] = float(hit.score)
            results.append(frame_record)

        return results
