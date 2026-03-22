from __future__ import annotations

from typing import Any, Sequence

import config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams


class QdrantClientWrapper:
    def __init__(
        self,
        url: str = config.QDRANT_URL,
        frame_collection_name: str = config.QDRANT_FRAME_COLLECTION,
        segment_collection_name: str = config.QDRANT_SEGMENT_COLLECTION,
        vector_size: int = config.OPENCLIP_VECTOR_SIZE,
    ) -> None:
        self.url = url
        self.frame_collection_name = frame_collection_name
        self.segment_collection_name = segment_collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url, check_compatibility=False)

    def init_collections(self, recreate: bool = False) -> None:
        self._init_collection(self.frame_collection_name, recreate=recreate)
        self._init_collection(self.segment_collection_name, recreate=recreate)

    def reset_collections(self) -> None:
        self.init_collections(recreate=True)

    def init_collection(self, recreate: bool = False) -> None:
        self.init_collections(recreate=recreate)

    def upsert_frame_point(
        self,
        frame_id: str,
        vector: Sequence[float],
    ) -> Any:
        return self._upsert_point(
            collection_name=self.frame_collection_name,
            point_id=frame_id,
            vector=vector,
            payload={"frame_id": frame_id},
        )

    def upsert_segment_point(
        self,
        segment_id: str,
        vector: Sequence[float],
    ) -> Any:
        return self._upsert_point(
            collection_name=self.segment_collection_name,
            point_id=segment_id,
            vector=vector,
            payload={"segment_id": segment_id},
        )

    def upsert_point(
        self,
        point_id: str,
        vector: Sequence[float],
        payload: dict[str, Any] | None = None,
    ) -> Any:
        return self._upsert_point(
            collection_name=self.frame_collection_name,
            point_id=point_id,
            vector=vector,
            payload=payload,
        )

    def _upsert_point(
        self,
        *,
        collection_name: str,
        point_id: str,
        vector: Sequence[float],
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self._validate_vector(vector)
        return self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload=payload or {},
                )
            ],
        )

    def upsert_points(self, points: list[PointStruct]) -> Any:
        for point in points:
            self._validate_vector(point.vector)
        return self.client.upsert(
            collection_name=self.frame_collection_name,
            wait=True,
            points=points,
        )

    def get_frame_point(self, frame_id: str, with_vector: bool = True) -> Any | None:
        return self._get_point(self.frame_collection_name, frame_id, with_vector=with_vector)

    def get_segment_point(self, segment_id: str, with_vector: bool = True) -> Any | None:
        return self._get_point(self.segment_collection_name, segment_id, with_vector=with_vector)

    def get_point(self, point_id: str, with_vector: bool = True) -> Any | None:
        return self.get_frame_point(point_id, with_vector=with_vector)

    def _get_point(self, collection_name: str, point_id: str, with_vector: bool = True) -> Any | None:
        records = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=with_vector,
        )
        if not records:
            return None
        return records[0]

    def get_points(self, point_ids: list[str], with_vector: bool = False) -> list[Any]:
        if not point_ids:
            return []
        return self.client.retrieve(
            collection_name=self.frame_collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=with_vector,
        )

    def search_frames(self, query_vector: Sequence[float], limit: int = 10) -> list[Any]:
        return self._search(self.frame_collection_name, query_vector, limit=limit)

    def search_segments(self, query_vector: Sequence[float], limit: int = 10) -> list[Any]:
        return self._search(self.segment_collection_name, query_vector, limit=limit)

    def search(self, query_vector: Sequence[float], limit: int = 10) -> list[Any]:
        return self.search_frames(query_vector, limit=limit)

    def _search(self, collection_name: str, query_vector: Sequence[float], limit: int = 10) -> list[Any]:
        self._validate_vector(query_vector)
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=collection_name,
                query_vector=list(query_vector),
                limit=limit,
                with_payload=True,
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def scroll(self, limit: int = 100) -> list[Any]:
        points, _ = self.client.scroll(
            collection_name=self.frame_collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return points

    def delete_point(self, point_id: str) -> Any:
        return self.delete_frame_point(point_id)

    def delete_frame_point(self, frame_id: str) -> Any:
        return self._delete_point(self.frame_collection_name, frame_id)

    def delete_segment_point(self, segment_id: str) -> Any:
        return self._delete_point(self.segment_collection_name, segment_id)

    def _delete_point(self, collection_name: str, point_id: str) -> Any:
        return self.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=[point_id]),
            wait=True,
        )

    def delete_points(self, point_ids: list[str]) -> Any:
        return self.client.delete(
            collection_name=self.frame_collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )

    def _init_collection(self, collection_name: str, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        if self.client.collection_exists(collection_name):
            collection = self.client.get_collection(collection_name)
            current_size = collection.config.params.vectors.size
            if current_size != self.vector_size:
                self.client.delete_collection(collection_name)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, got {len(vector)}."
            )
