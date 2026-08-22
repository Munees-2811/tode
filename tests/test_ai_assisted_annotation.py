"""
Unit tests for AI-Assisted Annotation workflow:
YOLO Detection → AI Suggestions → Human Verify (Accept, Edit, Delete, Accept All, Reject All, Confidence Threshold) → Final Annotation
"""
import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from models.annotation_model import BoundingBox, PolygonAnnotation, FrameAnnotation
from core.annotation_manager import AnnotationManager
from storage.label_storage import LabelStorage
from core.exporter import DatasetExporter


class TestAISuggestionModel:
    """Test FrameAnnotation suggestion lifecycle methods."""

    def test_suggestions_initialised_empty(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        assert ann.suggested_boxes == []
        assert ann.suggested_polygons == []
        assert not ann.is_annotated

    def test_add_and_clear_suggestions(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        b1 = BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, confidence=0.85)
        p1 = PolygonAnnotation(1, "cat", [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], confidence=0.90)

        ann.add_suggested_box(b1)
        ann.add_suggested_polygon(p1)

        assert len(ann.suggested_boxes) == 1
        assert len(ann.suggested_polygons) == 1
        # Suggestions do NOT count as confirmed annotation
        assert not ann.is_annotated

        ann.clear_suggestions()
        assert len(ann.suggested_boxes) == 0
        assert len(ann.suggested_polygons) == 0

    def test_accept_suggested_box(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        b1 = BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, confidence=0.85)
        ann.add_suggested_box(b1)

        accepted = ann.accept_suggested_box(0)
        assert accepted == b1
        assert len(ann.suggested_boxes) == 0
        assert len(ann.boxes) == 1
        assert ann.is_annotated

    def test_reject_suggested_box(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        b1 = BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, confidence=0.85)
        ann.add_suggested_box(b1)

        rejected = ann.reject_suggested_box(0)
        assert rejected == b1
        assert len(ann.suggested_boxes) == 0
        assert len(ann.boxes) == 0
        assert not ann.is_annotated

    def test_accept_all_suggested_boxes_with_confidence_filter(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        b1 = BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, confidence=0.90)
        b2 = BoundingBox(1, "cat", 0.3, 0.3, 0.1, 0.1, confidence=0.40)
        ann.add_suggested_box(b1)
        ann.add_suggested_box(b2)

        # Accept all with min_confidence = 0.5
        accepted = ann.accept_all_suggested_boxes(min_confidence=0.50)
        assert len(accepted) == 1
        assert accepted[0] == b1
        assert len(ann.boxes) == 1
        assert ann.boxes[0] == b1
        # b2 remains in suggested_boxes because its confidence < 0.50
        assert len(ann.suggested_boxes) == 1
        assert ann.suggested_boxes[0] == b2

    def test_reject_all_suggested_boxes(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        ann.add_suggested_box(BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, 0.8))
        ann.add_suggested_box(BoundingBox(1, "car", 0.1, 0.1, 0.2, 0.2, 0.7))

        ann.reject_all_suggested_boxes()
        assert len(ann.suggested_boxes) == 0
        assert len(ann.boxes) == 0

    def test_accept_and_reject_suggested_polygons(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        p1 = PolygonAnnotation(0, "road", [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)], confidence=0.95)
        p2 = PolygonAnnotation(1, "sky", [(0.0, 0.0), (1.0, 0.0), (1.0, 0.3)], confidence=0.30)
        ann.add_suggested_polygon(p1)
        ann.add_suggested_polygon(p2)

        accepted = ann.accept_suggested_polygon(0)
        assert accepted == p1
        assert len(ann.polygons) == 1
        assert ann.polygons[0] == p1

        ann.reject_all_suggested_polygons()
        assert len(ann.suggested_polygons) == 0

    def test_confidence_filtering(self):
        ann = FrameAnnotation(0, "frame_000000.png")
        b1 = BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, confidence=0.85)
        b2 = BoundingBox(1, "cat", 0.3, 0.3, 0.1, 0.1, confidence=0.35)
        ann.add_suggested_box(b1)
        ann.add_suggested_box(b2)

        filtered = ann.get_filtered_suggested_boxes(min_confidence=0.50)
        assert len(filtered) == 1
        assert filtered[0] == b1


class TestAnnotationManagerSuggestions:
    """Test AnnotationManager orchestration for AI suggestions."""

    @pytest.fixture
    def manager(self, tmp_path):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        v_loader   = MagicMock()
        v_loader.total_frames = 1
        v_loader.read_frame.return_value = dummy_frame
        f_extract  = MagicMock()
        frame_file = str(tmp_path / "frame_000000.png")
        f_extract.frame_path.return_value = frame_file
        f_extract.extract_single.return_value = (dummy_frame, frame_file)
        
        yolo       = MagicMock()
        b1 = BoundingBox(0, "person", 0.5, 0.5, 0.2, 0.2, confidence=0.88)
        yolo.annotate_frame.return_value = [b1]
        p1 = PolygonAnnotation(0, "building", [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4)], confidence=0.92)
        yolo.annotate_polygons_frame.return_value = [p1]

        f_store    = MagicMock()
        l_store    = MagicMock()

        mgr = AnnotationManager(v_loader, f_extract, yolo, f_store, l_store)
        mgr._read_frame_reliable = MagicMock(return_value=dummy_frame)
        mgr._annotations[0] = FrameAnnotation(0, frame_file)
        return mgr

    def test_auto_annotate_frame_creates_suggestions(self, manager):
        ann = manager.auto_annotate_frame(0)
        assert len(ann.suggested_boxes) == 1
        assert len(ann.boxes) == 0
        assert ann.suggested_boxes[0].class_name == "person"
        assert not ann.is_annotated

    def test_auto_annotate_polygons_frame_creates_suggestions(self, manager):
        ann = manager.auto_annotate_polygons_frame(0)
        assert len(ann.suggested_polygons) == 1
        assert len(ann.polygons) == 0
        assert ann.suggested_polygons[0].class_name == "building"

    def test_accept_suggestion_via_manager(self, manager):
        manager.auto_annotate_frame(0)
        accepted = manager.accept_suggestion(0, 0)
        assert accepted.class_name == "person"
        ann = manager.get_annotation(0)
        assert len(ann.boxes) == 1
        assert len(ann.suggested_boxes) == 0
        assert ann.is_annotated

    def test_reject_suggestion_via_manager(self, manager):
        manager.auto_annotate_frame(0)
        rejected = manager.reject_suggestion(0, 0)
        assert rejected.class_name == "person"
        ann = manager.get_annotation(0)
        assert len(ann.boxes) == 0
        assert len(ann.suggested_boxes) == 0

    def test_accept_and_reject_polygon_suggestion_via_manager(self, manager):
        manager.auto_annotate_polygons_frame(0)
        accepted = manager.accept_suggestion(0, 0, is_polygon=True)
        assert accepted.class_name == "building"
        ann = manager.get_annotation(0)
        assert len(ann.polygons) == 1
        assert len(ann.suggested_polygons) == 0

        # Now test reject
        manager.auto_annotate_polygons_frame(0)
        rejected = manager.reject_suggestion(0, 0, is_polygon=True)
        assert rejected.class_name == "building"
        assert len(ann.suggested_polygons) == 0


class TestStorageAndExporterIgnoresSuggestions:
    """Ensure suggestions are not written to disk or included in dataset exports."""

    @pytest.fixture(autouse=True)
    def tmp_labels(self, tmp_path, monkeypatch):
        import storage.label_storage as ls_mod
        import utils.config as cfg
        monkeypatch.setattr(cfg, "LABELS_DIR", str(tmp_path))
        monkeypatch.setattr(ls_mod, "LABELS_DIR", str(tmp_path))

    def test_label_storage_does_not_save_unaccepted_suggestions(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "test_sugg"
        storage.fmt        = "yolo"
        storage.base_dir   = str(tmp_path / "test_sugg")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        ann = FrameAnnotation(0, str(tmp_path / "frame_000000.png"))
        ann.add_suggested_box(BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, 0.85))

        # Save frame annotation
        storage.save(ann)
        label_file = storage._label_path(0, ext=".txt")

        # Label file should be empty because suggested_boxes are not confirmed
        assert os.path.exists(label_file)
        with open(label_file) as f:
            content = f.read().strip()
        assert content == ""

    def test_exporter_ignores_unaccepted_suggestions(self, tmp_path):
        frame_file = str(tmp_path / "frame_000000.png")
        # Write valid image file so DatasetExporter processes it
        cv2.imwrite(frame_file, np.zeros((100, 100, 3), dtype=np.uint8))

        ann = FrameAnnotation(0, frame_file)
        # 1 suggestion, 1 confirmed box
        ann.add_suggested_box(BoundingBox(0, "dog", 0.5, 0.5, 0.2, 0.2, 0.85))
        ann.add_box(BoundingBox(1, "car", 0.2, 0.2, 0.1, 0.1, 1.0))

        out_dir = str(tmp_path / "export_out")
        exporter = DatasetExporter(
            annotations={0: ann},
            class_names={0: "dog", 1: "car"},
            output_dir=out_dir,
        )

        res = exporter.export(fmt="yolo")
        assert res["images"] == 1
        assert res["labels"] == 1

        # Find exported txt file under labels directory
        label_dir = os.path.join(out_dir, "labels")
        if not os.path.exists(label_dir):
            label_dir = out_dir
        
        exported_txts = []
        for root, _, files in os.walk(label_dir):
            for f in files:
                if f.endswith(".txt") and f != "classes.txt":
                    exported_txts.append(os.path.join(root, f))
        
        assert len(exported_txts) == 1
        with open(exported_txts[0]) as f:
            lines = f.readlines()
        
        # Only the confirmed box "car" should be present (class_id 1)
        assert len(lines) == 1
        assert lines[0].startswith("1 ")
