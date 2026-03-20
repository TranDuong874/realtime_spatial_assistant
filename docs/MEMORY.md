# Memory Layer

This repo now includes a minimal memory subsystem for egocentric video recall:

- PostgreSQL is the source of truth for frames, clips, poses, and enrichment
- Qdrant stores only embeddings and lightweight references
- Query flow is always `Qdrant -> PostgreSQL`

## Setup

Start local services with Docker:

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

Environment variables:

```bash
export MEMORY_POSTGRES_DSN=postgresql://rsa:rsa@localhost:5432/rsa_memory
export MEMORY_QDRANT_URL=http://localhost:6333
export MEMORY_VECTOR_SIZE=512
```

## Schema

The Postgres schema lives at [memory/schema.sql](/home/tranduong/dev/realtime_spatial_assistant/memory/schema.sql).

Core tables:

- `clips`
- `poses`
- `frames`
- `frame_enrichments`

`frame_enrichments` is where you attach YOLO, segmentation, and OCR output once
those services are ready. Until then the placeholder pipeline writes empty lists.

## Usage

```python
from memory import ClipInput, FrameInput, MemorySystem, PoseInput

memory = MemorySystem.from_env()
memory.initialize()

clip = ClipInput(id="clip_42", start_time=1712345600, end_time=1712345650)
pose = PoseInput(id=1, tx=1.2, ty=-0.4, tz=0.8)
frame = FrameInput(
    id="frame_123",
    timestamp=1712345678,
    clip_id=clip.id,
    pose=pose,
    image_path="/data/frame_123.jpg",
)

memory.insert_clip(clip)
memory.insert_frame(frame, embedding=[0.0] * 512)

clip_hits = memory.search_clips(query_embedding=[0.0] * 512, limit=5)
records = memory.get_frames([hit.clip_id for hit in clip_hits])
```

If you have not written clip-level embeddings yet, `search_clips()` falls back to
the frame collection and groups the best matching frames by `clip_id`.

## MCP / LLM Flow

Your MCP server only needs three calls:

1. Convert the user query to an embedding
2. Call `MemorySystem.search_clips(...)`
3. Call `MemorySystem.get_frames(...)` for the returned clip ids

That gives the LLM structured frame recall with pose data attached.
