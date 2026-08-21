"""
core/base_detector.py
─────────────────────
Detector abstraction — every backend (Ultralytics, ONNX, future) implements
this interface so the rest of the app never imports a specific backend.

Breaking the interface free of ultralytics means:
  - .onnx weights load via onnxruntime (MIT) with zero AGPL surface
  - future backends (OpenVINO, TFLite, CoreML) drop in without UI changes
"""
from abc import ABC, abstractmethod

from models.annotation_model import BoundingBox, PolygonAnnotation


class BaseDetector(ABC):
    """Shared interface for every detection backend."""

    confidence: float = 0.45
    iou:        float = 0.45

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load weights from *model_path*.  Must be idempotent."""

    @abstractmethod
    def is_loaded(self) -> bool: ...

    # ── inference ─────────────────────────────────────────────────────────────
    @abstractmethod
    def detect(self, bgr_frame) -> list[BoundingBox]:
        """
        Run detection on a single BGR frame (numpy HWC uint8).
        Returns normalised BoundingBox list (x_center, y_center, w, h ∈ [0,1]).
        """

    def detect_polygons(self, bgr_frame) -> list[PolygonAnnotation]:
        """
        Run segmentation detection on a single BGR frame.
        Returns normalised PolygonAnnotation list.
        """
        boxes = self.detect(bgr_frame)
        from models.annotation_model import PolygonAnnotation
        img_h, img_w = bgr_frame.shape[:2] if bgr_frame is not None else (480, 640)
        polys: list[PolygonAnnotation] = []
        for b in boxes:
            x1 = max(0.0, (b.x_center - b.width / 2))
            y1 = max(0.0, (b.y_center - b.height / 2))
            x2 = min(1.0, (b.x_center + b.width / 2))
            y2 = min(1.0, (b.y_center + b.height / 2))
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            polys.append(PolygonAnnotation(
                class_id=b.class_id, class_name=b.class_name, points=pts, confidence=b.confidence
            ))
        return polys

    def detect_batch(self, bgr_frames: list) -> list[list[BoundingBox]]:
        """
        Optional batch API. Default implementation falls back to calling
        `detect` for each frame which preserves backwards compatibility.
        Backends can override this for faster batched inference.
        """
        return [self.detect(f) for f in bgr_frames]

    def detect_polygons_batch(self, bgr_frames: list) -> list[list[PolygonAnnotation]]:
        """Optional batch polygon API."""
        return [self.detect_polygons(f) if f is not None else [] for f in bgr_frames]

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Return {class_id: class_name} dict."""

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def backend_name(self) -> str:
        return self.__class__.__name__
