"""Global configuration constants."""
import os

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
FRAMES_DIR  = os.path.join(OUTPUT_DIR, "frames")
LABELS_DIR  = os.path.join(OUTPUT_DIR, "labels")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Default model — auto-downloaded by Ultralytics if the file is missing
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolo26x.pt")

# ── Frame extraction ──────────────────────────────────────────────────────────
DEFAULT_FPS_STEP = 1        # extract every N-th frame
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# ── Model catalogue ───────────────────────────────────────────────────────────
# Shown in the UI dropdown.
# Items ending in .onnx load via ONNXDetector (MIT, AGPL-free).
# Items without extension load via UltralyticsDetector (AGPL-3.0).

# Standard Ultralytics models — auto-downloaded on first use
_ULTRALYTICS_MODELS = [
    "yolo26x", "yolo26l", "yolo26m", "yolo26s", "yolo26n",
    "yolo11x", "yolo11l", "yolo11m", "yolo11s", "yolo11n",
    "yolov8x", "yolov8l", "yolov8m", "yolov8s", "yolov8n",
]

# Tode custom / branded models — ONNX files in weights/, AGPL-free at runtime.
# To add your own model:
#   1. Train YOLO: yolo train data=dataset.yaml model=yolo26n.pt
#   2. Export:     yolo export model=best.pt format=onnx imgsz=640
#   3. Copy:       cp best.onnx weights/todev1.onnx
#   4. Classes:    cp classes.json weights/todev1_classes.json
# Then add the entry below and it appears in the dropdown automatically.
_TODE_MODELS = [
    os.path.join(WEIGHTS_DIR, "todev1.onnx"),
]

# Full catalogue exposed to the UI (Tode models first)
YOLO_MODELS = (
    [m for m in _TODE_MODELS if os.path.isfile(m)]  # only show if file exists
    + _ULTRALYTICS_MODELS
)
YOLO_DEFAULT_MODEL = "yolo26x"

# Dedicated Segmentation Models catalogue (used specifically for Auto Polygon Segmentation)
_ULTRALYTICS_SEG_MODELS = [
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
    "yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg",
]

YOLO_SEG_MODELS = (
    [m for m in _TODE_MODELS if os.path.isfile(m) and "-seg" in m]
    + _ULTRALYTICS_SEG_MODELS
)
YOLO_DEFAULT_SEG_MODEL = "yolo11n-seg"

# ── Inference defaults ────────────────────────────────────────────────────────
YOLO_CONFIDENCE    = 0.45
YOLO_IOU_THRESHOLD = 0.45

# ── UI colours ────────────────────────────────────────────────────────────────
BG_DARK    = "#1e1e2e"
BG_PANEL   = "#2a2a3e"
ACCENT     = "#7c6af7"
TEXT_LIGHT = "#e0e0f0"
BOX_COLOR  = "#00ff88"          # manual bbox overlay colour

# ── Label storage format ──────────────────────────────────────────────────────
# "yolo" → .txt  (class cx cy w h — normalised)
LABEL_FORMAT = "yolo"

for _d in (OUTPUT_DIR, FRAMES_DIR, LABELS_DIR, WEIGHTS_DIR):
    os.makedirs(_d, exist_ok=True)
