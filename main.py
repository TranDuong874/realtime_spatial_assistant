from __future__ import annotations

from pathlib import Path
import time
import uuid

import cv2

import config
from pipeline import EvaluationStoragePipeline
from schema import FrameRecord
from services import OpenClipEmbeddingService, YoloDetectionService


def build_speed_metrics(
    source_fps: float,
    sample_every_n_frames: int,
    processed_count: int,
    elapsed_seconds: float,
) -> dict[str, float | bool]:
    incoming_fps = source_fps / max(1, sample_every_n_frames)
    processing_fps = 0.0 if elapsed_seconds <= 0 else processed_count / elapsed_seconds
    lag_seconds = 0.0 if incoming_fps <= 0 else max(0.0, elapsed_seconds - (processed_count / incoming_fps))
    return {
        "source_fps": source_fps,
        "incoming_fps": incoming_fps,
        "processing_fps": processing_fps,
        "speed_ratio": 0.0 if incoming_fps <= 0 else processing_fps / incoming_fps,
        "lag_seconds": lag_seconds,
        "is_falling_behind": processing_fps < incoming_fps,
    }


def run() -> None:
    if config.VIDEO_PATH == "/path/to/video.mp4":
        raise SystemExit("Set VIDEO_PATH in config.py first.")

    video_path = Path(config.VIDEO_PATH).expanduser().resolve()
    video_id = video_path.stem
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames_dir = Path("test_scripts/yolo_open_clip/frames").resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    yolo = YoloDetectionService()
    open_clip = OpenClipEmbeddingService()
    storage_pipeline = EvaluationStoragePipeline()
    storage_pipeline.initialize()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_index = 0
    processed = 0
    started_at = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % config.SAMPLE_EVERY_N_FRAMES != 0:
            frame_index += 1
            continue

        timestamp_ms = int(round((frame_index / fps) * 1000.0))
        frame_name = f"frame_{frame_index:06d}"
        frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_path}:{frame_index}"))
        frame_path = frames_dir / f"{frame_name}.jpg"
        cv2.imwrite(str(frame_path), frame)

        detections = yolo.detect_frame(frame)
        embedding = open_clip.embed_image(frame)
        frame_record = FrameRecord(
            frame_id=frame_id,
            frame_idx=frame_index,
            timestamp_ms=timestamp_ms,
            frame_path=str(frame_path),
            yolo_json=detections,
        )
        storage_pipeline.store_frame(
            frame_record,
            openclip_embedding=embedding,
            video_id=video_id,
        )

        processed += 1
        elapsed_seconds = time.perf_counter() - started_at
        metrics = build_speed_metrics(
            source_fps=fps,
            sample_every_n_frames=config.SAMPLE_EVERY_N_FRAMES,
            processed_count=processed,
            elapsed_seconds=elapsed_seconds,
        )
        print(
            "\r"
            f"stored={processed} "
            f"incoming_fps={metrics['incoming_fps']:.2f} "
            f"processing_fps={metrics['processing_fps']:.2f} "
            f"ratio={metrics['speed_ratio']:.2f} "
            f"lag_s={metrics['lag_seconds']:.2f} "
            f"behind={metrics['is_falling_behind']}",
            end="",
            flush=True,
        )
        frame_index += 1

        if config.MAX_FRAMES is not None and processed >= config.MAX_FRAMES:
            break

    capture.release()
    print()
    elapsed_seconds = time.perf_counter() - started_at
    metrics = build_speed_metrics(
        source_fps=fps,
        sample_every_n_frames=config.SAMPLE_EVERY_N_FRAMES,
        processed_count=processed,
        elapsed_seconds=elapsed_seconds,
    )
    print(
        "summary "
        f"source_fps={metrics['source_fps']:.2f} "
        f"incoming_fps={metrics['incoming_fps']:.2f} "
        f"processing_fps={metrics['processing_fps']:.2f} "
        f"ratio={metrics['speed_ratio']:.2f} "
        f"lag_s={metrics['lag_seconds']:.2f} "
        f"behind={metrics['is_falling_behind']}"
    )


if __name__ == "__main__":
    run()
