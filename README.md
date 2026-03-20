# realtime_spatial_assistant

This repo now has two layers:

- `slam`: Python bindings for the vendored `thirdparty/ORB_SLAM3` tree
- `memory`: a minimal Postgres + Qdrant recall layer for egocentric video memory

The intended architecture is:

```text
Client -> server -> SLAM-tagged frames -> enrichment -> embeddings -> recall
```

Storage responsibilities:

- PostgreSQL stores the real data: `clips`, `frames`, `poses`, `frame_enrichments`
- Qdrant stores only vectors and lightweight references like `frame_id` and `clip_id`
- Query flow is always `Qdrant -> PostgreSQL`

## Quick Start

Build the vendored native dependencies and the Python extension:

```bash
cd realtime_spatial_assistant
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

Run the import smoke test:

```bash
cd realtime_spatial_assistant
python3 test_slam_import.py
```

Install the Python dependencies for the memory layer:

```bash
python3 -m pip install -e .
```

Start local Postgres and Qdrant:

```bash
docker run -d \
  --name rsa-postgres \
  -e POSTGRES_USER=rsa \
  -e POSTGRES_PASSWORD=rsa \
  -e POSTGRES_DB=rsa_memory \
  -p 5432:5432 \
  postgres:16

docker run -d \
  --name rsa-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:v1.13.4
```

Configure the memory layer:

```bash
export MEMORY_POSTGRES_DSN=postgresql://rsa:rsa@localhost:5432/rsa_memory
export MEMORY_QDRANT_URL=http://localhost:6333
export MEMORY_VECTOR_SIZE=512
```

## Package

```python
import slam

system = slam.System(
    "thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt",
    "path/to/settings.yaml",
    slam.Sensor.IMU_STEREO,
    use_viewer=False,
)
```

Exports:

- `slam.System`
- `slam.Sensor`
- `slam.ImuMeasurement`

### `memory`

```python
from memory import ClipInput, FrameInput, MemorySystem, PoseInput

memory = MemorySystem.from_env()
memory.initialize()

clip = ClipInput(id="clip_42", start_time=1712345600, end_time=1712345650)
pose = PoseInput(id=1, tx=0.1, ty=0.2, tz=0.3)
frame = FrameInput(
    id="frame_123",
    timestamp=1712345678,
    clip_id=clip.id,
    pose=pose,
    image_path="/tmp/frame_123.jpg",
)

memory.insert_clip(clip)
memory.insert_frame(frame, embedding=[0.0] * 512)

clip_hits = memory.search_clips(query_embedding=[0.0] * 512, limit=5)
records = memory.get_frames([hit.clip_id for hit in clip_hits])
```

The placeholder perception pipeline is already wired into storage. Right now it
writes empty YOLO / segmentation / OCR results into Postgres, and later you can
swap in real model-backed services without changing the schema or query path.

### `main_pipeline.py`

For early development, `main_pipeline.py` now supports an MP4-first ingest path:

```bash
export MEMORY_POSTGRES_DSN=postgresql://rsa:rsa@localhost:5432/rsa_memory
export MEMORY_QDRANT_URL=http://localhost:6333
export MEMORY_VECTOR_SIZE=512

python3 main_pipeline.py /path/to/video.mp4 \
  --sample-every-n-frames 30 \
  --max-frames 100 \
  --frame-output-dir ./sampled_frames
```

What it does:

- reads a raw MP4 with OpenCV
- samples frames at a fixed stride
- creates a simple development embedding for each frame
- writes frame rows into Postgres
- writes frame vectors into Qdrant
- computes one clip vector as the mean of sampled frame vectors

This gives you a working end-to-end memory ingest path before SLAM, YOLO, OCR,
or segmentation are integrated.

## Docs

- `docs/API.md`: Python API reference for `slam`
- `docs/BUILD.md`: build prerequisites, build flow, rebuild notes
- `docs/MEMORY.md`: Postgres + Qdrant setup and memory query flow

## Notes

- The build defaults to `MAKE_JOBS=2` because ORB-SLAM3 is memory-heavy.
- The extension binary is generated at `slam/_orbslam3*.so`.
- The vendored ORB-SLAM3 native build products are treated as generated files and are ignored by `.gitignore`.
- In the memory layer, Postgres is the source of truth and Qdrant is only the vector index.
