"""
core/yolo_annotator.py
───────────────────────
Public-facing detector facade.  All other modules continue to use:

    from core.yolo_annotator import YOLOAnnotator

The internal backend is selected automatically from the file extension:
    .pt   → UltralyticsDetector  (ultralytics, AGPL-3.0)
    .onnx → ONNXDetector         (onnxruntime,  MIT — AGPL-free)

Passing a bare model name ("yolo26x", "todev1") defaults to Ultralytics.
The public interface is unchanged — annotation_manager and the UI are
not aware of which backend is active.
"""
import os
import threading

from core.base_detector import BaseDetector
from models.annotation_model import BoundingBox
from utils.config import YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_MODEL_PATH
from utils.logger import get_logger

log = get_logger("core.YOLOAnnotator")


def _make_detector(model_path: str, confidence: float, iou: float) -> BaseDetector:
    """Pick the right backend based on file extension and model architecture."""
    norm_path = model_path.lower()
    if norm_path.endswith(".onnx"):
        from core.detectors.onnx_detector import ONNXDetector
        log.info(f"Backend selected: ONNX Runtime  ({model_path})")
        d = ONNXDetector(confidence=confidence, iou=iou)
    elif "rtdetr" in norm_path or "rt-detr" in norm_path:
        from core.detectors.rtdetr_detector import RTDETRDetector
        log.info(f"Backend selected: RT-DETR  ({model_path})")
        d = RTDETRDetector(confidence=confidence, iou=iou)
    else:
        from core.detectors.ultralytics_detector import UltralyticsDetector
        log.info(f"Backend selected: Ultralytics  ({model_path})")
        d = UltralyticsDetector(confidence=confidence, iou=iou)
    return d


class YOLOAnnotator:
    """
    Thread-safe detector facade.
    Drop-in replacement for the previous direct-YOLO implementation —
    every caller keeps using the same .load(), .reload(), .annotate_frame(),
    .class_names, .confidence, and .iou interface.
    """

    def __init__(
        self,
        model_path: str   = YOLO_MODEL_PATH,
        confidence: float = YOLO_CONFIDENCE,
        iou:        float = YOLO_IOU_THRESHOLD,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._iou        = iou
        self._detector: BaseDetector = _make_detector(model_path, confidence, iou)
        self._lock = threading.Lock()
        log.debug(
            f"YOLOAnnotator created — conf={confidence}, iou={iou}, "
            f"path={model_path}"
        )

    # ── confidence / iou pass-through ─────────────────────────────────────────
    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, val: float):
        self._confidence = val
        self._detector.confidence = val

    @property
    def iou(self) -> float:
        return self._iou

    @iou.setter
    def iou(self, val: float):
        self._iou = val
        self._detector.iou = val

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self):
        with self._lock:
            if self._detector.is_loaded():
                return
            self._detector.load(self._model_path)

    def reload(self, model_name_or_path: str):
        """
        Swap model at runtime. If backend changed (YOLO ↔ RT-DETR ↔ ONNX), a
        new backend is created transparently.
        """
        path = model_name_or_path
        if not os.path.exists(path) and not path.endswith((".pt", ".onnx")):
            path = path + ".pt"

        with self._lock:
            # Rebuild backend if detector type/backend changed
            current_type = type(self._detector)
            temp_detector = _make_detector(path, self._confidence, self._iou)
            if type(temp_detector) is not current_type:
                self._detector = temp_detector
            else:
                self._detector.confidence = self._confidence
                self._detector.iou        = self._iou

            # Clear cached seg model so it re-loads for the new weights
            if hasattr(self._detector, '_seg_model'):
                self._detector._seg_model = None
                self._detector._seg_model_path = ""

            self._model_path = path

        self._detector.load(path)
        log.info(
            f"Model reloaded — path={path}  "
            f"backend={self._detector.backend_name}"
        )

    def is_loaded(self) -> bool:
        return self._detector.is_loaded()

    # ── inference ─────────────────────────────────────────────────────────────
    def annotate_frame(self, bgr_frame) -> list[BoundingBox]:
        self.load()
        log.debug(
            f"Running detection — conf={self._confidence}, iou={self._iou}, "
            f"backend={self._detector.backend_name}"
        )
        boxes = self._detector.detect(bgr_frame)
        log.info(
            f"Detection complete — {len(boxes)} object(s)  "
            f"[{self._detector.backend_name}]"
        )
        return boxes

    def annotate_frames(self, bgr_frames: list) -> list[list[BoundingBox]]:
        """
        Batch annotation API. Delegates to backend `detect_batch` when
        available which allows backends to optimise batched inference.
        """
        self.load()
        log.debug(
            f"Running batched detection — {len(bgr_frames)} frames, "
            f"conf={self._confidence}, iou={self._iou}, "
            f"backend={self._detector.backend_name}"
        )
        try:
            boxes_list = self._detector.detect_batch(bgr_frames)
        except Exception:
            # Fallback: run single-frame detect for each
            boxes_list = [self._detector.detect(f) if f is not None else [] for f in bgr_frames]
        total = sum(len(b) for b in boxes_list)
        log.info(
            f"Batched detection complete — {total} object(s)  "
            f"[{self._detector.backend_name}]"
        )
        return boxes_list

    def annotate_polygons_frame(self, bgr_frame) -> list:
        self.load()
        log.debug(
            f"Running polygon segmentation — conf={self._confidence}, iou={self._iou}, "
            f"backend={self._detector.backend_name}"
        )
        polys = self._detector.detect_polygons(bgr_frame)
        log.info(
            f"Polygon segmentation complete — {len(polys)} polygon(s)  "
            f"[{self._detector.backend_name}]"
        )
        return polys

    def annotate_polygons_frames(self, bgr_frames: list) -> list[list]:
        self.load()
        try:
            polys_list = self._detector.detect_polygons_batch(bgr_frames)
        except Exception:
            polys_list = [self._detector.detect_polygons(f) if f is not None else [] for f in bgr_frames]
        total = sum(len(p) for p in polys_list)
        log.info(
            f"Batched polygon segmentation complete — {total} polygon(s)  "
            f"[{self._detector.backend_name}]"
        )
        return polys_list

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    def class_names(self) -> dict[int, str]:
        return self._detector.class_names

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def backend_name(self) -> str:
        return self._detector.backend_name
