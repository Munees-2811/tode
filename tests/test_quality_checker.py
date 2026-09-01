"""Unit tests for QualityChecker engine."""
import pytest

from core.quality_checker import QualityChecker
from models.annotation_model import BoundingBox, FrameAnnotation, PolygonAnnotation


@pytest.fixture
def class_map():
    return {0: "person", 1: "car", 2: "dog"}


def test_clean_annotations(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.2, height=0.3))
    ann.add_polygon(PolygonAnnotation(class_id=1, class_name="car", points=[(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)]))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.is_clean
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.valid_boxes == 1
    assert report.valid_polygons == 1


def test_missing_labels(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=0, class_name="", x_center=0.5, y_center=0.5, width=0.2, height=0.2))
    ann.add_polygon(PolygonAnnotation(class_id=1, class_name="   ", points=[(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)]))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.error_count == 2
    assert any("empty/missing class label" in err.message for err in report.errors)


def test_empty_annotation():
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.is_annotated = True

    checker = QualityChecker({0: ann}, {0: "person"})
    report = checker.run()

    assert report.warning_count == 1
    assert "marked as annotated but contains no" in report.warnings[0].message


def test_invalid_bounding_boxes(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.0, height=0.2))
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.2, height=-0.1))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.error_count == 2
    assert all("invalid dimensions" in err.message for err in report.errors)


def test_boxes_outside_image(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.0, y_center=0.5, width=0.4, height=0.4))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.warning_count >= 1
    assert any("extends outside image" in w.message for w in report.warnings)


def test_very_small_and_large_boxes(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    # Tiny box
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.01, height=0.01))
    # Huge box
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.98, height=0.98))

    checker = QualityChecker({0: ann}, class_map, min_box_area=0.001, max_box_area=0.90)
    report = checker.run()

    assert report.warning_count == 2
    assert any("very small" in w.message for w in report.warnings)
    assert any("very large" in w.message for w in report.warnings)


def test_invalid_polygon_points(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_polygon(PolygonAnnotation(class_id=0, class_name="person", points=[(0.1, 0.1), (0.2, 0.2)]))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.error_count == 1
    assert "fewer than 3 points" in report.errors[0].message


def test_self_intersecting_polygon(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    # Bowtie polygon: (0,0) -> (1,1) -> (1,0) -> (0,1)
    ann.add_polygon(PolygonAnnotation(
        class_id=0, class_name="person",
        points=[(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)],
    ))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.warning_count >= 1
    assert any("self-intersecting" in w.message for w in report.warnings)


def test_duplicate_boxes(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.3, height=0.3))
    ann.add_box(BoundingBox(class_id=0, class_name="person", x_center=0.5, y_center=0.5, width=0.3, height=0.3))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.warning_count == 1
    assert "overlap heavily" in report.warnings[0].message


def test_unknown_class_id(class_map):
    ann = FrameAnnotation(frame_index=0, frame_path="dummy.png")
    ann.add_box(BoundingBox(class_id=99, class_name="spaceship", x_center=0.5, y_center=0.5, width=0.3, height=0.3))

    checker = QualityChecker({0: ann}, class_map)
    report = checker.run()

    assert report.error_count == 1
    assert "not in the known class list" in report.errors[0].message
