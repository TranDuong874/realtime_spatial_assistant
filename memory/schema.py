from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class FrameInput:
    id: str
    timestamp: int
    image_path: str | None = None

__all__ = [
    "FrameInput",
]
