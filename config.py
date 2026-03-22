POSTGRES_DSN = "postgresql://rsa:rsa@localhost:5432/rsa_memory"
QDRANT_URL = "http://localhost:6333"
QDRANT_FRAME_COLLECTION = "frame_openclip"
QDRANT_ACTION_COLLECTION = "action_segment"
QDRANT_WINDOW_COLLECTION = "window_openclip"
OPENCLIP_VECTOR_SIZE = 768
VECTOR_SIZE = OPENCLIP_VECTOR_SIZE

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
OPENCLIP_RETRIEVAL_FRAME_STRIDE_FRAMES = 30
OPENCLIP_WINDOW_SECONDS = 60.0
OPENCLIP_WINDOW_OVERLAP = 0.5

PADDLEOCR_LANGUAGE = "en"
PADDLEOCR_OCR_VERSION = "PP-OCRv5"
PADDLEOCR_USE_GPU = True
PADDLEOCR_USE_TEXTLINE_ORIENTATION = False
PADDLEOCR_TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
PADDLEOCR_TEXT_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
PADDLEOCR_TEXT_DET_THRESH = 0.4
PADDLEOCR_TEXT_DET_BOX_THRESH = 0.7
PADDLEOCR_TEXT_REC_SCORE_THRESH = 0.85
PADDLEOCR_MIN_TEXT_CHARS = 2

SLOWFAST_REPO_PATH = "thirdparty/epic-kitchens-slowfast"
SLOWFAST_CHECKPOINT_PATH = "models/slowfast/SlowFast.pyth"
SLOWFAST_DEVICE = "cuda"
SLOWFAST_CLIP_NUM_FRAMES = 32
SLOWFAST_CLIP_TARGET_FPS = 30.0
SLOWFAST_CLIP_STRIDE_FRAMES = 16
SLOWFAST_BATCH_SIZE = 1
SLOWFAST_MAX_CLIPS = None

ACTIONFORMER_REPO_PATH = "thirdparty/actionformer_release"
ACTIONFORMER_DEVICE = "cuda"
ACTIONFORMER_DEFAULT_FPS = 30.0
ACTIONFORMER_FEAT_STRIDE = 16
ACTIONFORMER_FEAT_NUM_FRAMES = 32
ACTIONFORMER_INPUT_DIM = 2304
ACTIONFORMER_EPIC_VERB_CONFIG_PATH = "thirdparty/actionformer_release/configs/epic_slowfast_verb.yaml"
ACTIONFORMER_EPIC_NOUN_CONFIG_PATH = "thirdparty/actionformer_release/configs/epic_slowfast_noun.yaml"
ACTIONFORMER_EPIC_VERB_CHECKPOINT_PATH = "models/actionformer/epic_slowfast_verb_reproduce/epoch_020.pth.tar"
ACTIONFORMER_EPIC_NOUN_CHECKPOINT_PATH = "models/actionformer/epic_slowfast_noun_reproduce/epoch_020.pth.tar"

ACTIONFORMER_CONFIG_PATH = "thirdparty/actionformer_release/configs/ego4d_egovlp.yaml"
ACTIONFORMER_CHECKPOINT_PATH = "models/actionformer/ego4d_egovlp_reproduce/epoch_010.pth.tar"
TBN_CHECKPOINT_PATH = "models/tbn/TBN-epic-kitchens-100.pth"

# Backup if ViT-L-14 runs out of VRAM:
# OPEN_CLIP_MODEL_NAME = "ViT-B-16"
# OPEN_CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"

# Future large deployed model:
# OPEN_CLIP_MODEL_NAME = "ViT-SO400M-14-SigLIP"
# OPEN_CLIP_PRETRAINED = "webli"
