"""
core/detectors/rtdetr_detector.py
───────────────────────────────────
Detector backed by RT-DETR (Real-Time DEtection TRansformer) via Ultralytics.

Supports RT-DETR models such as rtdetr-l, rtdetr-x, rtdetr-resnet50,
rtdetr-resnet101, or custom trained RT-DETR weights.
"""
import os

from core.base_detector import BaseDetector
from models.annotation_model import BoundingBox
from utils.logger import get_logger

log = get_logger("core.detectors.RTDETRDetector")


class RTDETRDetector(BaseDetector):
    """
    Wraps Ultralytics RT-DETR models for transformer-based real-time object detection.
    """

    def __init__(self, confidence: float = 0.45, iou: float = 0.45):
        self.confidence = confidence
        self.iou        = iou
        self._model     = None
        self._model_path: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        if self._model is not None and self._model_path == model_path:
            return
        self._load_weights(model_path)

    def _load_weights(self, weights: str) -> None:
        if not os.path.exists(weights) and not weights.endswith((".pt", ".onnx", ".engine")):
            weights = weights + ".pt"

        log.info(f"[RT-DETR] loading: {weights}")
        try:
            try:
                from ultralytics import RTDETR
                self._model = RTDETR(weights)
            except (ImportError, AttributeError):
                from ultralytics import YOLO
                log.info(f"[RT-DETR] RTDETR class not directly available, using YOLO loader for '{weights}'")
                self._model = YOLO(weights)

            self._model_path = weights
            log.info(
                f"[RT-DETR] ready — "
                f"{len(self.class_names)} classes"
            )
        except Exception as exc:
            log.error(f"[RT-DETR] load failed for '{weights}': {exc}", exc_info=True)
            self._model = None
            self._model_path = ""
            raise

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── inference ─────────────────────────────────────────────────────────────
    def detect(self, bgr_frame) -> list[BoundingBox]:
        if not self.is_loaded() or bgr_frame is None:
            return []
        try:
            results = self._model.predict(
                source  = bgr_frame,
                conf    = self.confidence,
                iou     = self.iou,
                imgsz   = 640,
                verbose = False,
            )
        except Exception as exc:
            log.error(f"[RT-DETR] inference error: {exc}", exc_info=True)
            return []

        boxes: list[BoundingBox] = []
        img_h, img_w = bgr_frame.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return []

        for result in results:
            names = result.names or self.class_names
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Normalize to [0, 1]
                x_center = ((x1 + x2) / 2.0) / img_w
                y_center = ((y1 + y2) / 2.0) / img_h
                width    = (x2 - x1) / img_w
                height   = (y2 - y1) / img_h

                boxes.append(BoundingBox(
                    class_id   = cls_id,
                    class_name = cls_name,
                    x_center   = x_center,
                    y_center   = y_center,
                    width      = width,
                    height     = height,
                    confidence = conf,
                ))
        log.info(f"[RT-DETR] {len(boxes)} detection(s)")
        return boxes

    def detect_batch(self, bgr_frames: list) -> list[list[BoundingBox]]:
        """
        Run batched inference using the Ultralytics `predict` API with a list of frames.
        """
        if not self.is_loaded() or not bgr_frames:
            return [[] for _ in bgr_frames]

        valid_frames = [f for f in bgr_frames if f is not None]
        if not valid_frames:
            return [[] for _ in bgr_frames]

        try:
            results = self._model.predict(
                source  = bgr_frames,
                conf    = self.confidence,
                iou     = self.iou,
                imgsz   = 640,
                verbose = False,
            )
        except Exception as exc:
            log.error(f"[RT-DETR] batched inference error: {exc}", exc_info=True)
            return [self.detect(f) if f is not None else [] for f in bgr_frames]

        batch_boxes: list[list[BoundingBox]] = []
        for i, result in enumerate(results):
            frame = bgr_frames[i]
            if frame is None:
                batch_boxes.append([])
                continue
            img_h, img_w = frame.shape[:2]
            if img_h <= 0 or img_w <= 0:
                batch_boxes.append([])
                continue

            frame_boxes: list[BoundingBox] = []
            names = result.names or self.class_names
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    conf     = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x_center = ((x1 + x2) / 2.0) / img_w
                    y_center = ((y1 + y2) / 2.0) / img_h
                    width    = (x2 - x1) / img_w
                    height   = (y2 - y1) / img_h

                    frame_boxes.append(BoundingBox(
                        class_id   = cls_id,
                        class_name = cls_name,
                        x_center   = x_center,
                        y_center   = y_center,
                        width      = width,
                        height     = height,
                        confidence = conf,
                    ))
            batch_boxes.append(frame_boxes)

        return batch_boxes

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            return {}
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    @property
    def backend_name(self) -> str:
        return "RT-DETR (Ultralytics)"
