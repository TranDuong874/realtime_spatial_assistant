POSTGRES_DSN = "postgresql://rsa:rsa@localhost:5432/rsa_memory"
QDRANT_URL = "http://localhost:6333"
QDRANT_FRAME_COLLECTION = "frame_embeddings"
VECTOR_SIZE = 768

VIDEO_PATH = "/home/tranduong/dev/realtime_spatial_assistant/test_data/loc5_script4_seq2_rec1/AriaEverydayActivities_1.0.0_loc5_script4_seq2_rec1_preview_rgb.mp4"
SAMPLE_EVERY_N_FRAMES = 2
MAX_FRAMES = None

YOLO_MODEL_PATH = "models/yolo26n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = None

OPEN_CLIP_MODEL_NAME = "ViT-L-14"
OPEN_CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"
OPEN_CLIP_CACHE_DIR = "models"
OPEN_CLIP_DEVICE = "cuda"

# Backup if ViT-L-14 runs out of VRAM:
# OPEN_CLIP_MODEL_NAME = "ViT-B-16"
# OPEN_CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"

# Future large deployed model:
# OPEN_CLIP_MODEL_NAME = "ViT-SO400M-14-SigLIP"
# OPEN_CLIP_PRETRAINED = "webli"
