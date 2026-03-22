from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Frame(Base):
    __tablename__ = "frames"

    frame_id: Mapped[str] = mapped_column(String, primary_key=True)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    frame_path: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_json: Mapped[list[dict[str, Any]] | dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    yolo_json: Mapped[list[dict[str, Any]] | dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    slam_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    segment_starts: Mapped[list["Segment"]] = relationship(
        back_populates="start_frame",
        foreign_keys="Segment.start_frame_id",
    )
    segment_ends: Mapped[list["Segment"]] = relationship(
        back_populates="end_frame",
        foreign_keys="Segment.end_frame_id",
    )


class Segment(Base):
    __tablename__ = "segments"

    segment_id: Mapped[str] = mapped_column(String, primary_key=True)
    start_frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("frames.frame_id", ondelete="CASCADE"),
        nullable=False,
    )
    end_frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("frames.frame_id", ondelete="CASCADE"),
        nullable=False,
    )
    start_frame_idx: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    end_frame_idx: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    start_s: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    end_s: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    verb_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    noun_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_text: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rep_frame_start_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("frames.frame_id", ondelete="SET NULL"),
        nullable=True,
    )
    rep_frame_mid_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("frames.frame_id", ondelete="SET NULL"),
        nullable=True,
    )
    rep_frame_end_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("frames.frame_id", ondelete="SET NULL"),
        nullable=True,
    )

    start_frame: Mapped[Frame] = relationship(back_populates="segment_starts", foreign_keys=[start_frame_id])
    end_frame: Mapped[Frame] = relationship(back_populates="segment_ends", foreign_keys=[end_frame_id])
