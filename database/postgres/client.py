from __future__ import annotations

from typing import Any

import config
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

from database.postgres.models import Base, Frame, Segment


class PostgresClient:
    def __init__(self, dsn: str = config.POSTGRES_DSN) -> None:
        self.dsn = dsn
        self.engine = create_engine(self.dsn, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def reset_db(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        Base.metadata.create_all(self.engine)

    def upsert_frame(
        self,
        frame_id: str,
        frame_idx: int,
        timestamp_ms: int,
        frame_path: str,
        ocr_text: str | None = None,
        ocr_json: list[dict[str, Any]] | dict[str, Any] | None = None,
        yolo_json: list[dict[str, Any]] | dict[str, Any] | None = None,
        slam_json: dict[str, Any] | None = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(Frame.__table__).values(
                    frame_id=frame_id,
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    frame_path=frame_path,
                    ocr_text=ocr_text,
                    ocr_json=ocr_json,
                    yolo_json=yolo_json,
                    slam_json=slam_json,
                ).on_conflict_do_update(
                    index_elements=[Frame.frame_id],
                    set_={
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "frame_path": frame_path,
                        "ocr_text": ocr_text,
                        "ocr_json": ocr_json,
                        "yolo_json": yolo_json,
                        "slam_json": slam_json,
                    },
                )
            )

    def upsert_segment(
        self,
        segment_id: str,
        start_frame_id: str,
        end_frame_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        start_s: float,
        end_s: float,
        action_text: str,
        score: float,
        verb_label: str | None = None,
        noun_label: str | None = None,
        rep_frame_start_id: str | None = None,
        rep_frame_mid_id: str | None = None,
        rep_frame_end_id: str | None = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(Segment.__table__).values(
                    segment_id=segment_id,
                    start_frame_id=start_frame_id,
                    end_frame_id=end_frame_id,
                    start_frame_idx=start_frame_idx,
                    end_frame_idx=end_frame_idx,
                    start_s=start_s,
                    end_s=end_s,
                    verb_label=verb_label,
                    noun_label=noun_label,
                    action_text=action_text,
                    score=score,
                    rep_frame_start_id=rep_frame_start_id,
                    rep_frame_mid_id=rep_frame_mid_id,
                    rep_frame_end_id=rep_frame_end_id,
                ).on_conflict_do_update(
                    index_elements=[Segment.segment_id],
                    set_={
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id,
                        "start_frame_idx": start_frame_idx,
                        "end_frame_idx": end_frame_idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "verb_label": verb_label,
                        "noun_label": noun_label,
                        "action_text": action_text,
                        "score": score,
                        "rep_frame_start_id": rep_frame_start_id,
                        "rep_frame_mid_id": rep_frame_mid_id,
                        "rep_frame_end_id": rep_frame_end_id,
                    },
                )
            )

    def get_frame(self, frame_id: str) -> Frame | None:
        with self.session_factory() as session:
            return session.get(Frame, frame_id)

    def get_segment(self, segment_id: str) -> Segment | None:
        with self.session_factory() as session:
            return session.get(Segment, segment_id)

    def get_frames(self, frame_ids: list[str]) -> list[Frame]:
        if not frame_ids:
            return []

        with self.session_factory() as session:
            return session.scalars(
                select(Frame).where(Frame.frame_id.in_(frame_ids)).order_by(Frame.frame_idx.asc())
            ).all()

    def list_frames(self, limit: int = 100) -> list[Frame]:
        with self.session_factory() as session:
            return session.scalars(
                select(Frame).order_by(Frame.frame_idx.asc()).limit(limit)
            ).all()

    def delete_frame(self, frame_id: str) -> None:
        with self.session_factory.begin() as session:
            frame = session.get(Frame, frame_id)
            if frame is not None:
                session.delete(frame)
