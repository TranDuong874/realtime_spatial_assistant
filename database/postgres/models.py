from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    frame_path: Mapped[str] = mapped_column(Text, nullable=False)
    frame_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    yolo: Mapped["FrameYolo | None"] = relationship(
        back_populates="frame",
        cascade="all, delete-orphan",
        uselist=False,
    )


class FrameYolo(Base):
    __tablename__ = "frame_yolo"

    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("frames.id", ondelete="CASCADE"),
        primary_key=True,
    )
    detections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    frame: Mapped[Frame] = relationship(back_populates="yolo")
