# realtime_spatial_assistant

Current focus:

- `services/slowfast.py`: EPIC-KITCHENS SlowFast clip feature extraction
- `services/actionformer.py`: EPIC-KITCHENS ActionFormer verb/noun segment inference
- `schema/action.py`: typed clip, window, segment, and action-sequence schemas
- `pipeline/action_pipeline.py`: action post-processing, merging, and sequence assembly
- `services/open_clip.py`: image embeddings
- `services/yolo.py`: frame detections
- `pipeline/` + `database/`: Postgres and Qdrant storage path

Current development pipeline:

```text
video
  -> 32-frame SlowFast clips, stride 16, target 30 FPS
  -> 2304-d clip features
  -> rolling ActionFormer verb/noun windows
  -> merged action segments
```

## Quick Start

Create a virtual environment and install the repo:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you want the frame-storage pipeline, start local Postgres and Qdrant:

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

## Installed Models

Current local checkpoints in active use:

```text
models/yolo26n.pt
models/slowfast/SlowFast.pyth
models/actionformer/epic_slowfast_verb_reproduce/epoch_020.pth.tar
models/actionformer/epic_slowfast_noun_reproduce/epoch_020.pth.tar
```

Supporting repos in active use:

```text
thirdparty/epic-kitchens-slowfast
thirdparty/actionformer_release
datasets/epic-kitchens-100-annotations
```

## SlowFast Setup

The local SlowFast service uses the EPIC-KITCHENS model and clip config:

- 32 input frames
- stride 16 frames
- target FPS 30
- output embedding size 2304

Relevant config in `config.py`:

```python
SLOWFAST_REPO_PATH = "thirdparty/epic-kitchens-slowfast"
SLOWFAST_CHECKPOINT_PATH = "models/slowfast/SlowFast.pyth"
SLOWFAST_CLIP_NUM_FRAMES = 32
SLOWFAST_CLIP_TARGET_FPS = 30.0
SLOWFAST_CLIP_STRIDE_FRAMES = 16
```

## ActionFormer Setup

The local ActionFormer service is now a thin model adapter. It accepts an `ActionWindowInput`,
runs inference, and returns typed raw segment predictions. Segment filtering, label resolution,
merging, and action-sequence building live in `pipeline/action_pipeline.py`.

The local ActionFormer service loads the EPIC-KITCHENS verb and noun models from:

```python
ACTIONFORMER_EPIC_VERB_CONFIG_PATH = "thirdparty/actionformer_release/configs/epic_slowfast_verb.yaml"
ACTIONFORMER_EPIC_NOUN_CONFIG_PATH = "thirdparty/actionformer_release/configs/epic_slowfast_noun.yaml"
ACTIONFORMER_EPIC_VERB_CHECKPOINT_PATH = "models/actionformer/epic_slowfast_verb_reproduce/epoch_020.pth.tar"
ACTIONFORMER_EPIC_NOUN_CHECKPOINT_PATH = "models/actionformer/epic_slowfast_noun_reproduce/epoch_020.pth.tar"
```

The service consumes SlowFast features, not raw video.

Expected feature contract:

- feature dim 2304
- feat stride 16
- feat window 32 frames
- default FPS 30

## Test Runners

Frame pipeline:

```bash
python3 main.py
```

SlowFast-only video feature test:

```bash
python3 test_scripts/slowfast_video_test.py
```

Streaming SlowFast + ActionFormer console test with annotated output video:

```bash
PYTHONUNBUFFERED=1 ./.venv/bin/python test_scripts/slowfast_actionformer_video_test.py
```

Useful flags:

```bash
--video-path /abs/path/to/video.mp4
--score-threshold 0.25
--min-duration-seconds 0.8
--max-segments-per-window 4
--max-clips 60
--write-video
```

## Output Artifacts

The streaming test writes outputs under the chosen `--output-dir`:

- `summary.json`
- `window_results.json`
- `merged_segments.json`
- `action_sequence.txt`
- `annotated_actions.mp4` when `--write-video` is enabled

## Notes

- SlowFast is currently the runtime bottleneck in the streaming test.
- ActionFormer inference is feature-based temporal localization over SlowFast embeddings.
- EPIC label names are resolved from `datasets/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv` and `EPIC_100_noun_classes.csv`.
