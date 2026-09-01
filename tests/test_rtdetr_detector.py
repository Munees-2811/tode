"""Unit tests for RTDETRDetector and YOLOAnnotator RT-DETR routing."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.detectors.onnx_detector import ONNXDetector
from core.detectors.rtdetr_detector import RTDETRDetector
from core.detectors.ultralytics_detector import UltralyticsDetector
from core.yolo_annotator import YOLOAnnotator
from models.annotation_model import BoundingBox


class TestRTDETRDetectorBasics:
    def test_initial_state(self):
        det = RTDETRDetector(confidence=0.5, iou=0.4)
        assert not det.is_loaded()
        assert det.confidence == 0.5
        assert det.iou == 0.4
        assert det.class_names == {}
        assert det.backend_name == "RT-DETR (Ultralytics)"

    def test_detect_returns_empty_when_unloaded(self):
        det = RTDETRDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert det.detect(frame) == []
        assert det.detect(None) == []
        assert det.detect_batch([frame, None]) == [[], []]

    def test_detect_with_mock_model(self):
        det = RTDETRDetector(confidence=0.45, iou=0.45)
        # Mock model predict
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bicycle"}

        mock_box = MagicMock()
        mock_box.cls = [np.array(0)]
        mock_box.conf = [np.array(0.92)]
        mock_box.xyxy = [np.array([100.0, 50.0, 300.0, 250.0])]

        mock_result = MagicMock()
        mock_result.names = {0: "person", 1: "bicycle"}
        mock_result.boxes = [mock_box]

        mock_model.predict.return_value = [mock_result]
        det._model = mock_model
        det._model_path = "rtdetr-l.pt"

        assert det.is_loaded()
        assert det.class_names == {0: "person", 1: "bicycle"}

        frame = np.zeros((500, 1000, 3), dtype=np.uint8)  # h=500, w=1000
        boxes = det.detect(frame)

        assert len(boxes) == 1
        box = boxes[0]
        assert isinstance(box, BoundingBox)
        assert box.class_id == 0
        assert box.class_name == "person"
        assert box.confidence == pytest.approx(0.92)
        # x_center = (100+300)/2 / 1000 = 0.2
        assert box.x_center == pytest.approx(0.2)
        # y_center = (50+250)/2 / 500 = 0.3
        assert box.y_center == pytest.approx(0.3)
        # width = (300-100) / 1000 = 0.2
        assert box.width == pytest.approx(0.2)
        # height = (250-50) / 500 = 0.4
        assert box.height == pytest.approx(0.4)

    def test_detect_batch_with_mock_model(self):
        det = RTDETRDetector()
        mock_model = MagicMock()
        mock_model.names = {0: "dog"}

        mock_box = MagicMock()
        mock_box.cls = [np.array(0)]
        mock_box.conf = [np.array(0.85)]
        mock_box.xyxy = [np.array([0.0, 0.0, 200.0, 200.0])]

        mock_result1 = MagicMock()
        mock_result1.names = {0: "dog"}
        mock_result1.boxes = [mock_box]

        mock_result2 = MagicMock()
        mock_result2.names = {0: "dog"}
        mock_result2.boxes = []

        mock_model.predict.return_value = [mock_result1, mock_result2]
        det._model = mock_model

        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)

        results = det.detect_batch([frame1, frame2])
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0].class_name == "dog"
        assert len(results[1]) == 0

    def test_inference_error_handling(self):
        det = RTDETRDetector()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU out of memory")
        det._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should gracefully return empty list instead of crashing
        assert det.detect(frame) == []


class TestYOLOAnnotatorRTDETRRouting:
    def test_rtdetr_models_route_to_rtdetr_detector(self):
        for name in ["rtdetr-l.pt", "rtdetr-x", "rtdetr-resnet50.pt", "weights/custom_rtdetr.pt"]:
            ann = YOLOAnnotator(model_path=name)
            assert isinstance(ann._detector, RTDETRDetector)
            assert ann.backend_name == "RT-DETR (Ultralytics)"

    def test_yolo_models_route_to_ultralytics_detector(self):
        ann = YOLOAnnotator(model_path="yolo26x.pt")
        assert isinstance(ann._detector, UltralyticsDetector)

    def test_onnx_models_route_to_onnx_detector(self, tmp_path):
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert isinstance(ann._detector, ONNXDetector)

    def test_dynamic_reload_swapping(self, tmp_path, monkeypatch):
        # Patch load so it doesn't try downloading actual weights during tests
        monkeypatch.setattr(RTDETRDetector, "load", lambda self, p: None)
        monkeypatch.setattr(UltralyticsDetector, "load", lambda self, p: None)
        monkeypatch.setattr(ONNXDetector, "load", lambda self, p: None)

        ann = YOLOAnnotator(model_path="yolo26x.pt")
        assert isinstance(ann._detector, UltralyticsDetector)

        ann.reload("rtdetr-l")
        assert isinstance(ann._detector, RTDETRDetector)
        assert ann.model_path == "rtdetr-l.pt"

        onnx_path = str(tmp_path / "model.onnx")
        ann.reload(onnx_path)
        assert isinstance(ann._detector, ONNXDetector)

        ann.reload("rtdetr-x")
        assert isinstance(ann._detector, RTDETRDetector)
