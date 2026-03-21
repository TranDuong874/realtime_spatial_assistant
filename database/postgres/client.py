from __future__ import annotations

from typing import Any

import config
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

from database.postgres.models import Base, Frame, FrameYolo


class PostgresClient:
    def __init__(self, dsn: str = config.POSTGRES_DSN) -> None:
        self.dsn = dsn
        self.engine = create_engine(self.dsn, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def upsert_frame(
        self,
        frame_id: str,
        timestamp_ms: int,
        frame_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(Frame.__table__).values(
                    id=frame_id,
                    timestamp_ms=timestamp_ms,
                    frame_path=frame_path,
                    metadata=metadata or {},
                ).on_conflict_do_update(
                    index_elements=[Frame.id],
                    set_={
                        "timestamp_ms": timestamp_ms,
                        "frame_path": frame_path,
                        "metadata": metadata or {},
                    },
                )
            )

    def upsert_frame_yolo(
        self,
        frame_id: str,
        detections: list[dict[str, Any]],
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(FrameYolo.__table__).values(
                    frame_id=frame_id,
                    detections=detections,
                ).on_conflict_do_update(
                    index_elements=[FrameYolo.frame_id],
                    set_={"detections": detections},
                )
            )

    def get_frame(self, frame_id: str) -> Frame | None:
        with self.session_factory() as session:
            return session.get(Frame, frame_id)

    def get_frame_yolo(self, frame_id: str) -> FrameYolo | None:
        with self.session_factory() as session:
            return session.get(FrameYolo, frame_id)

    def get_frames(self, frame_ids: list[str]) -> list[Frame]:
        if not frame_ids:
            return []

        with self.session_factory() as session:
            return session.scalars(
                select(Frame).where(Frame.id.in_(frame_ids)).order_by(Frame.timestamp_ms.asc())
            ).all()

    def list_frames(self, limit: int = 100) -> list[Frame]:
        with self.session_factory() as session:
            return session.scalars(
                select(Frame).order_by(Frame.timestamp_ms.asc()).limit(limit)
            ).all()

    def delete_frame(self, frame_id: str) -> None:
        with self.session_factory.begin() as session:
            frame = session.get(Frame, frame_id)
            if frame is not None:
                session.delete(frame)
