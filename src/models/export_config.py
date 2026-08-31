"""
models/export_config.py
─────────────────────────
Configuration model for a dataset export operation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportConfig:
    """All settings needed to drive an export operation."""

    output_dir:      str
    format:          str = "yolo_split"
    split:           bool = True
    train_ratio:     float = 0.70
    test_ratio:      float = 0.10
    seed:            int = 42
    use_random_seed: bool = False
    copy_images:     bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.train_ratio <= 1.0):
            raise ValueError("train_ratio must be in (0, 1]")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")

    def val_ratio(self) -> float:
        return max(0.0, 1.0 - self.train_ratio - self.test_ratio)
