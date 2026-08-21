"""
core/detectors/ultralytics_detector.py
────────────────────────────────────────
Detector backed by the Ultralytics library (YOLO26 / YOLO11 / YOLOv8).

LICENCE NOTE: Ultralytics is AGPL-3.0.  Any code that imports this module
inherits that obligation.  For closed-source or SaaS deployments use
ONNXDetector instead — it carries no AGPL surface.
"""
import os
import re

import numpy as np

from core.base_detector import BaseDetector
from models.annotation_model import BoundingBox, PolygonAnnotation
from utils.logger import get_logger

log = get_logger("core.detectors.UltralyticsDetector")


def _box_to_polygon(frame, x1: int, y1: int, x2: int, y2: int) -> list[tuple[float, float]]:
    """
    Extract an object contour from the bounding-box crop.
    Uses adaptive thresholding + morphological cleanup for much better
    results than the old simple Otsu approach.
    """
    import cv2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1

    rect_pts = [
        (float(x1 / w), float(y1 / h)),
        (float(x2 / w), float(y1 / h)),
        (float(x2 / w), float(y2 / h)),
        (float(x1 / w), float(y2 / h)),
    ]
    if bw < 5 or bh < 5:
        return rect_pts

    # Pad crop slightly for better edge detection
    pad = max(3, min(bw, bh) // 10)
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w - 1, x2 + pad)
    cy2 = min(h - 1, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # --- Strategy 1: GrabCut (best quality, works within bounding box) ---
        gc_mask = np.zeros(crop.shape[:2], np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        gc_rect = (pad, pad, crop.shape[1] - 2 * pad, crop.shape[0] - 2 * pad)
        if gc_rect[2] > 4 and gc_rect[3] > 4:
            cv2.grabCut(crop, gc_mask, gc_rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                crop_area = bw * bh
                if cv2.contourArea(cnt) > crop_area * 0.05:
                    # Use tighter approximation for smoother polygons
                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    pts = [(float((cx1 + pt[0][0]) / w), float((cy1 + pt[0][1]) / h)) for pt in approx]
                    if len(pts) >= 3:
                        return pts

        # --- Strategy 2: Adaptive threshold + edge detection fallback ---
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive threshold handles varying illumination better than Otsu
        thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        # Also try Otsu and combine
        _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Edge detection for objects with subtle boundaries
        edges = cv2.Canny(blur, 30, 100)
        edges = cv2.dilate(edges, None, iterations=2)

        # Combine all masks
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, edges)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 30:
                epsilon = 0.008 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = [(float((cx1 + pt[0][0]) / w), float((cy1 + pt[0][1]) / h)) for pt in approx]
                if len(pts) >= 3:
                    return pts
    except Exception as exc:
        log.debug(f"_box_to_polygon fallback failed: {exc}")

    return rect_pts


def _seg_model_path(det_path: str) -> str | None:
    """
    Derive the segmentation model name from a detection model path.
    e.g. 'yolo26x' → 'yolo26x-seg', 'yolo11m.pt' → 'yolo11m-seg.pt'
    """
    base = os.path.basename(det_path)
    # Already a seg model
    if "-seg" in base:
        return det_path
    # .pt file on disk
    if base.endswith(".pt"):
        return det_path.replace(".pt", "-seg.pt")
    # Bare model name (auto-downloaded by ultralytics)
    if not "." in base:
        return det_path + "-seg"
    return None


class UltralyticsDetector(BaseDetector):
    """Wraps ultralytics.YOLO.  Accepts any .pt model name or path."""

    def __init__(self, confidence: float = 0.45, iou: float = 0.45):
        self.confidence = confidence
        self.iou        = iou
        self._model     = None
        self._seg_model = None          # separate seg model for polygon mode
        self._model_path: str = ""
        self._seg_model_path: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        if self._model is not None and self._model_path == model_path:
            return
        self._load_weights(model_path)

    def _load_weights(self, weights: str) -> None:
        from ultralytics import YOLO  # import isolated here — AGPL surface

        if not os.path.exists(weights) and not weights.endswith(".pt") and not weights.endswith(".onnx"):
            weights = weights + ".pt"

        log.info(f"[Ultralytics] loading: {weights}")
        try:
            self._model      = YOLO(weights)
            self._model_path = weights
            log.info(
                f"[Ultralytics] ready — "
                f"{len(self._model.names)} classes"
            )
        except Exception as exc:
            log.warning(f"[Ultralytics] load failed for '{weights}': {exc}")
            # If a -seg variant (like yolo26x-seg.pt) failed to download, fallback to base model
            if "-seg" in weights and not os.path.exists(weights):
                fallback = weights.replace("-seg", "")
                log.info(f"[Ultralytics] fallback to base model: {fallback}")
                self._model      = YOLO(fallback)
                self._model_path = fallback
                log.info(f"[Ultralytics] ready (fallback) — {len(self._model.names)} classes")
            else:
                raise

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── inference ─────────────────────────────────────────────────────────────
    def _ensure_seg_model(self) -> bool:
        """
        Lazily load the segmentation variant of the current model.
        Returns True if a seg model is available.
        """
        if self._seg_model is not None:
            return True
        seg_path = _seg_model_path(self._model_path)
        if seg_path is None:
            return False
        try:
            from ultralytics import YOLO
            log.info(f"[Ultralytics] loading seg model: {seg_path}")
            self._seg_model = YOLO(seg_path)
            self._seg_model_path = seg_path
            log.info(
                f"[Ultralytics] seg model ready — "
                f"{len(self._seg_model.names)} classes"
            )
            return True
        except Exception as exc:
            log.warning(f"[Ultralytics] seg model unavailable ({seg_path}): {exc}")
            self._seg_model = None
            return False

    # ── inference ─────────────────────────────────────────────────────────────
    def detect(self, bgr_frame) -> list[BoundingBox]:
        if not self.is_loaded():
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
            log.error(f"[Ultralytics] inference error: {exc}", exc_info=True)
            return []

        boxes: list[BoundingBox] = []
        for result in results:
            img_h, img_w = bgr_frame.shape[:2]
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(BoundingBox(
                    class_id   = cls_id,
                    class_name = cls_name,
                    x_center   = ((x1 + x2) / 2) / img_w,
                    y_center   = ((y1 + y2) / 2) / img_h,
                    width      = (x2 - x1)        / img_w,
                    height     = (y2 - y1)        / img_h,
                    confidence = conf,
                ))
        log.info(f"[Ultralytics] {len(boxes)} detection(s)")
        return boxes

    def detect_batch(self, bgr_frames: list) -> list[list[BoundingBox]]:
        """
        Run batched inference using the Ultralytics `predict` API which accepts
        a list of images. Returns a list of per-image BoundingBox lists.
        """
        if not self.is_loaded():
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
            log.error(f"[Ultralytics] batched inference error: {exc}", exc_info=True)
            return [[] for _ in bgr_frames]

        all_boxes: list[list[BoundingBox]] = []
        for result in results:
            img_h, img_w = result.orig_shape if hasattr(result, 'orig_shape') else (None, None)
            if img_h is None:
                # Fallback to first input shape if orig_shape not present
                img_h, img_w = bgr_frames[0].shape[:2]
            boxes: list[BoundingBox] = []
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(BoundingBox(
                    class_id   = cls_id,
                    class_name = cls_name,
                    x_center   = ((x1 + x2) / 2) / img_w,
                    y_center   = ((y1 + y2) / 2) / img_h,
                    width      = (x2 - x1)        / img_w,
                    height     = (y2 - y1)        / img_h,
                    confidence = conf,
                ))
            all_boxes.append(boxes)
        return all_boxes

    def detect_polygons(self, bgr_frame) -> list[PolygonAnnotation]:
        if not self.is_loaded():
            return []

        # ── prefer the loaded model if it is already a segmentation model (-seg) ──
        if "-seg" in os.path.basename(self._model_path):
            model = self._model
            use_seg = True
            model_label = "seg"
        else:
            use_seg = self._ensure_seg_model()
            model = self._seg_model if use_seg else self._model
            model_label = "seg" if use_seg else "det"

        try:
            predict_kwargs = dict(
                source       = bgr_frame,
                conf         = self.confidence,
                iou          = self.iou,
                imgsz        = 640,
                verbose      = False,
            )
            # retina_masks gives full-resolution masks (not down-scaled)
            if use_seg:
                predict_kwargs["retina_masks"] = True
            results = model.predict(**predict_kwargs)
        except Exception as exc:
            log.error(f"[Ultralytics] polygon inference error ({model_label}): {exc}", exc_info=True)
            # If seg model failed, retry with detection model
            if use_seg:
                log.info("[Ultralytics] falling back to detection model for polygons")
                try:
                    results = self._model.predict(
                        source  = bgr_frame,
                        conf    = self.confidence,
                        iou     = self.iou,
                        imgsz   = 640,
                        verbose = False,
                    )
                    use_seg = False
                except Exception as exc2:
                    log.error(f"[Ultralytics] fallback also failed: {exc2}", exc_info=True)
                    return []
            else:
                return []

        polys: list[PolygonAnnotation] = []
        for result in results:
            img_h, img_w = bgr_frame.shape[:2]
            has_masks = hasattr(result, "masks") and result.masks is not None
            xyn_list = result.masks.xyn if has_masks else []
            for i, box in enumerate(result.boxes):
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf     = float(box.conf[0])
                if has_masks and i < len(xyn_list) and len(xyn_list[i]) >= 3:
                    pts = [(float(x), float(y)) for x, y in xyn_list[i]]
                else:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pts = _box_to_polygon(bgr_frame, int(x1), int(y1), int(x2), int(y2))
                if len(pts) >= 3:
                    polys.append(PolygonAnnotation(
                        class_id   = cls_id,
                        class_name = cls_name,
                        points     = pts,
                        confidence = conf,
                    ))
        log.info(f"[Ultralytics] {len(polys)} polygon detection(s) [{model_label}]")
        return polys

    def detect_polygons_batch(self, bgr_frames: list) -> list[list[PolygonAnnotation]]:
        return [self.detect_polygons(f) if f is not None else [] for f in bgr_frames]

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names if self._model else {}

    @property
    def backend_name(self) -> str:
        return "Ultralytics"
