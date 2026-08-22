"""
tests/test_dataset_split.py
───────────────────────────
Unit tests for automatic Train / Validation / Test dataset split export.
"""
import os

import pytest
import yaml

from core.exporter import DatasetExporter
from models.annotation_model import BoundingBox, FrameAnnotation, PolygonAnnotation
from models.export_config import ExportConfig


@pytest.fixture
def mock_dataset(tmp_path):
    """Generate 10 dummy annotated frames with images on disk."""
    frames_dir = tmp_path / "src_frames"
    frames_dir.mkdir()

    annotations = {}
    for i in range(10):
        img_file = frames_dir / f"frame_{i:04d}.jpg"
        img_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01")  # dummy JPG bytes

        ann = FrameAnnotation(frame_index=i, frame_path=str(img_file))
        ann.add_box(
            BoundingBox(
                class_id=i % 2,
                class_name="cat" if (i % 2 == 0) else "dog",
                x_center=0.5,
                y_center=0.5,
                width=0.4,
                height=0.4,
            )
        )
        if i % 3 == 0:
            ann.add_polygon(
                PolygonAnnotation(
                    class_id=i % 2,
                    class_name="cat" if (i % 2 == 0) else "dog",
                    points=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)],
                )
            )
        annotations[i] = ann

    class_names = {0: "cat", 1: "dog"}
    return annotations, class_names


def test_dataset_split_ratios_and_structure(tmp_path, mock_dataset):
    annotations, class_names = mock_dataset
    out_dir = str(tmp_path / "dataset_export")

    exporter = DatasetExporter(annotations, class_names, out_dir)
    res = exporter.export(
        fmt="yolo_split",
        split=True,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        seed=42,
    )

    assert res["format"] == "yolo_split"
    assert res["total_images"] == 10
    assert res["train_images"] == 7
    assert res["val_images"] == 2
    assert res["test_images"] == 1

    # Verify directory structure
    for split_folder in ("train", "val", "test"):
        assert os.path.isdir(os.path.join(out_dir, split_folder, "images"))
        assert os.path.isdir(os.path.join(out_dir, split_folder, "labels"))

    # Verify classes.txt and data.yaml are not created by default
    assert not os.path.exists(os.path.join(out_dir, "classes.txt"))
    assert not os.path.exists(os.path.join(out_dir, "data.yaml"))

    # When export_yaml=True, data.yaml is created
    exporter.export(
        fmt="yolo_split",
        split=True,
        export_yaml=True,
        export_classes=True,
    )
    yaml_path = os.path.join(out_dir, "data.yaml")
    assert os.path.isfile(yaml_path)

    with open(yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["train"] == "train/images"
    assert data_cfg["val"] == "val/images"
    assert data_cfg["test"] == "test/images"
    assert data_cfg["nc"] == 2
    assert data_cfg["names"] == {0: "cat", 1: "dog"}


def test_dataset_split_no_duplicate_or_omitted_images(tmp_path, mock_dataset):
    annotations, class_names = mock_dataset
    out_dir = str(tmp_path / "dataset_export_unique")

    exporter = DatasetExporter(annotations, class_names, out_dir)
    exporter.export(
        fmt="yolo_split",
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        seed=123,
    )

    exported_files = set()
    for sname in ("train", "val", "test"):
        img_dir = os.path.join(out_dir, sname, "images")
        files = os.listdir(img_dir)
        for f in files:
            assert f not in exported_files, f"Duplicate file found: {f}"
            exported_files.add(f)

    assert len(exported_files) == 10


def test_dataset_split_reproducible_seed(tmp_path, mock_dataset):
    annotations, class_names = mock_dataset
    out_dir1 = str(tmp_path / "export_seed_1")
    out_dir2 = str(tmp_path / "export_seed_2")

    exp1 = DatasetExporter(annotations, class_names, out_dir1)
    exp1.export(fmt="yolo_split", seed=99, use_random_seed=False)

    exp2 = DatasetExporter(annotations, class_names, out_dir2)
    exp2.export(fmt="yolo_split", seed=99, use_random_seed=False)

    train1_imgs = sorted(os.listdir(os.path.join(out_dir1, "train", "images")))
    train2_imgs = sorted(os.listdir(os.path.join(out_dir2, "train", "images")))

    assert train1_imgs == train2_imgs


def test_export_config_defaults():
    cfg = ExportConfig(output_dir="/tmp/export")
    assert cfg.train_ratio == 0.70
    assert cfg.val_ratio() == pytest.approx(0.20)
    assert cfg.test_ratio == 0.10
    assert cfg.seed == 42
