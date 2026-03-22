from __future__ import annotations

from typing import Any

import config
from sqlalchemy import create_engine, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

from database.postgres.models import Base, Clip, Frame, Segment, SegmentClip, SegmentFrame


class PostgresClient:
    def __init__(self, dsn: str = config.POSTGRES_DSN) -> None:
        self.dsn = dsn
        self.engine = create_engine(self.dsn, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def reset_db(self) -> None:
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def upsert_frame(
        self,
        frame_id: str,
        frame_idx: int,
        timestamp_ms: int,
        frame_path: str,
        ocr_text: str | None = None,
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
                    yolo_json=yolo_json,
                    slam_json=slam_json,
                ).on_conflict_do_update(
                    index_elements=[Frame.frame_id],
                    set_={
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "frame_path": frame_path,
                        "ocr_text": ocr_text,
                        "yolo_json": yolo_json,
                        "slam_json": slam_json,
                    },
                )
            )

    def upsert_clip(
        self,
        clip_id: str,
        start_frame_id: str,
        end_frame_id: str,
        start_s: float,
        end_s: float,
        feature_path: str,
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(Clip.__table__).values(
                    clip_id=clip_id,
                    start_frame_id=start_frame_id,
                    end_frame_id=end_frame_id,
                    start_s=start_s,
                    end_s=end_s,
                    feature_path=feature_path,
                ).on_conflict_do_update(
                    index_elements=[Clip.clip_id],
                    set_={
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id,
                        "start_s": start_s,
                        "end_s": end_s,
                        "feature_path": feature_path,
                    },
                )
            )

    def upsert_segment(
        self,
        segment_id: str,
        start_frame_id: str,
        end_frame_id: str,
        start_s: float,
        end_s: float,
        action_text: str,
        score: float,
        verb_label: str | None = None,
        noun_label: str | None = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(Segment.__table__).values(
                    segment_id=segment_id,
                    start_frame_id=start_frame_id,
                    end_frame_id=end_frame_id,
                    start_s=start_s,
                    end_s=end_s,
                    verb_label=verb_label,
                    noun_label=noun_label,
                    action_text=action_text,
                    score=score,
                ).on_conflict_do_update(
                    index_elements=[Segment.segment_id],
                    set_={
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id,
                        "start_s": start_s,
                        "end_s": end_s,
                        "verb_label": verb_label,
                        "noun_label": noun_label,
                        "action_text": action_text,
                        "score": score,
                    },
                )
            )

    def replace_segment_frames(
        self,
        segment_id: str,
        frame_refs: list[dict[str, str]],
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(delete(SegmentFrame).where(SegmentFrame.segment_id == segment_id))
            if frame_refs:
                session.execute(
                    SegmentFrame.__table__.insert(),
                    [
                        {
                            "segment_id": segment_id,
                            "frame_id": item["frame_id"],
                            "role": item["role"],
                        }
                        for item in frame_refs
                    ],
                )

    def replace_segment_clips(
        self,
        segment_id: str,
        clip_ids: list[str],
    ) -> None:
        with self.session_factory.begin() as session:
            session.execute(delete(SegmentClip).where(SegmentClip.segment_id == segment_id))
            if clip_ids:
                session.execute(
                    SegmentClip.__table__.insert(),
                    [
                        {
                            "segment_id": segment_id,
                            "clip_id": clip_id,
                        }
                        for clip_id in clip_ids
                    ],
                )

    def get_frame(self, frame_id: str) -> Frame | None:
        with self.session_factory() as session:
            return session.get(Frame, frame_id)

    def get_clip(self, clip_id: str) -> Clip | None:
        with self.session_factory() as session:
            return session.get(Clip, clip_id)

    def get_segment(self, segment_id: str) -> Segment | None:
        with self.session_factory() as session:
            return session.get(Segment, segment_id)

    def get_segment_frames(self, segment_id: str) -> list[SegmentFrame]:
        with self.session_factory() as session:
            return session.scalars(
                select(SegmentFrame).where(SegmentFrame.segment_id == segment_id)
            ).all()

    def get_segment_clips(self, segment_id: str) -> list[SegmentClip]:
        with self.session_factory() as session:
            return session.scalars(
                select(SegmentClip).where(SegmentClip.segment_id == segment_id)
            ).all()

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
