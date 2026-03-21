from __future__ import annotations

from typing import Any, Sequence

import config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams


class QdrantClientWrapper:
    def __init__(
        self,
        url: str = config.QDRANT_URL,
        collection_name: str = config.QDRANT_FRAME_COLLECTION,
        vector_size: int = config.VECTOR_SIZE,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url, check_compatibility=False)

    def init_collection(self, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if self.client.collection_exists(self.collection_name):
            collection = self.client.get_collection(self.collection_name)
            current_size = collection.config.params.vectors.size
            if current_size != self.vector_size:
                self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert_point(
        self,
        point_id: str,
        vector: Sequence[float],
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self._validate_vector(vector)
        return self.client.upsert(
            collection_name=self.collection_name,
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
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )

    def get_point(self, point_id: str, with_vector: bool = True) -> Any | None:
        records = self.client.retrieve(
            collection_name=self.collection_name,
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
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=with_vector,
        )

    def search(self, query_vector: Sequence[float], limit: int = 10) -> list[Any]:
        self._validate_vector(query_vector)
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=limit,
            with_payload=True,
        )

    def scroll(self, limit: int = 100) -> list[Any]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return points

    def delete_point(self, point_id: str) -> Any:
        return self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
            wait=True,
        )

    def delete_points(self, point_ids: list[str]) -> Any:
        return self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, got {len(vector)}."
            )
