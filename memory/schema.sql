CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    start_time BIGINT NOT NULL,
    end_time BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS poses (
    id BIGINT PRIMARY KEY,
    tx DOUBLE PRECISION NOT NULL,
    ty DOUBLE PRECISION NOT NULL,
    tz DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    pose_id BIGINT NOT NULL REFERENCES poses(id) ON DELETE RESTRICT,
    image_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_frames_clip_id ON frames (clip_id);
CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames (timestamp);

CREATE TABLE IF NOT EXISTS frame_enrichments (
    frame_id TEXT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
    detected_objects JSONB NOT NULL DEFAULT '[]'::jsonb,
    segments JSONB NOT NULL DEFAULT '[]'::jsonb,
    ocr_text JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb
);
