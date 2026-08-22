"""Tests for advanced polygon segmentation features and auto-annotation."""
import numpy as np

from core.base_detector import BaseDetector
from models.annotation_model import FrameAnnotation, PolygonAnnotation


def test_polygon_geometry_area_and_perimeter():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    poly = PolygonAnnotation(class_id=0, class_name="box", points=pts)

    assert abs(poly.area(1, 1) - 1.0) < 1e-5
    assert abs(poly.perimeter(1, 1) - 4.0) < 1e-5

    assert abs(poly.area(640, 480) - (640 * 480)) < 1e-3
    assert abs(poly.perimeter(640, 480) - (2 * 640 + 2 * 480)) < 1e-3


def test_polygon_contains_point():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    poly = PolygonAnnotation(class_id=1, class_name="triangle", points=pts)

    assert poly.contains_point(0.5, 0.2) is True
    assert poly.contains_point(0.9, 0.9) is False
    assert poly.contains_point(0.0, 1.0) is False


def test_polygon_add_edit_and_remove():
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.jpg")
    poly = PolygonAnnotation(class_id=0, class_name="car", points=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.4)])
    ann.add_polygon(poly)

    assert len(ann.polygons) == 1
    assert ann.is_annotated is True

    # Edit vertex
    ann.polygons[0].points[1] = (0.5, 0.1)
    assert ann.polygons[0].points[1] == (0.5, 0.1)

    # Remove polygon
    ann.remove_polygon(0)
    assert len(ann.polygons) == 0
    assert ann.is_annotated is False


def test_polygon_copy_between_frames():
    f0 = FrameAnnotation(frame_index=0, frame_path="frame_0.jpg")
    f1 = FrameAnnotation(frame_index=1, frame_path="frame_1.jpg")

    poly = PolygonAnnotation(class_id=2, class_name="person", points=[(0.2, 0.2), (0.3, 0.2), (0.3, 0.5), (0.2, 0.5)])
    f0.add_polygon(poly)

    copied_polys = [
        PolygonAnnotation(
            class_id=p.class_id,
            class_name=p.class_name,
            points=list(p.points),
            confidence=p.confidence,
        )
        for p in f0.polygons
    ]
    for cp in copied_polys:
        f1.add_polygon(cp)

    assert len(f1.polygons) == 1
    assert f1.polygons[0].class_name == "person"
    assert f1.polygons[0].points == [(0.2, 0.2), (0.3, 0.2), (0.3, 0.5), (0.2, 0.5)]


class MockPolygonDetector(BaseDetector):
    def load(self, model_path: str) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def detect(self, bgr_frame):
        from models.annotation_model import BoundingBox
        return [BoundingBox(class_id=0, class_name="cat", x_center=0.5, y_center=0.5, width=0.4, height=0.4)]

    def detect_polygons(self, bgr_frame):
        return [PolygonAnnotation(class_id=0, class_name="cat", points=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])]

    @property
    def class_names(self):
        return {0: "cat"}


def test_polygon_auto_annotation():
    detector = MockPolygonDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    polys = detector.detect_polygons(dummy_frame)
    assert len(polys) == 1
    assert polys[0].class_name == "cat"
    assert len(polys[0].points) == 4
