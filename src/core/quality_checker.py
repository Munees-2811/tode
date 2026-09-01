"""
core/quality_checker.py
────────────────────────
Read-only annotation quality validation engine.

Scans all FrameAnnotation objects and produces a structured QualityReport
with categorised issues (errors, warnings, valid counts).

Does NOT modify any annotation data — purely observational.
"""
from dataclasses import dataclass, field
from enum import Enum

from models.annotation_model import FrameAnnotation
from utils.logger import get_logger

log = get_logger("core.QualityChecker")


# ── Data structures ──────────────────────────────────────────────────────────

class Severity(Enum):
    ERROR   = "error"
    WARNING = "warning"
    INFO    = "info"


@dataclass
class QualityIssue:
    """A single quality problem found in the annotations."""
    frame_index:  int
    severity:     Severity
    message:      str
    annotation_type: str = ""    # "box", "polygon", "classification", "frame"
    annotation_index: int = -1   # index within the frame's list (-1 = frame-level)

    @property
    def icon(self) -> str:
        return {
            Severity.ERROR:   "❌",
            Severity.WARNING: "⚠️",
            Severity.INFO:    "✅",
        }.get(self.severity, "●")


@dataclass
class QualityReport:
    """Aggregate result of a quality check run."""
    issues:          list[QualityIssue] = field(default_factory=list)
    total_frames:    int = 0
    annotated_frames: int = 0
    total_boxes:     int = 0
    total_polygons:  int = 0
    valid_boxes:     int = 0
    valid_polygons:  int = 0

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0 and self.warning_count == 0


# ── Checker engine ───────────────────────────────────────────────────────────

class QualityChecker:
    """
    Stateless checker — instantiate, call ``run()``, read the report.

    Parameters
    ----------
    annotations : dict[int, FrameAnnotation]
        The annotation dictionary from AnnotationManager._annotations.
    class_names : dict[int, str]
        Known class ID → name map (from the YOLO model).
    min_box_area : float
        Normalised area threshold below which a box is "very small" (default 0.001 = 0.1%).
    max_box_area : float
        Normalised area threshold above which a box is "very large" (default 0.90 = 90%).
    overlap_iou_threshold : float
        IoU threshold above which two boxes are considered duplicates (default 0.95).
    """

    def __init__(
        self,
        annotations: dict[int, FrameAnnotation],
        class_names: dict[int, str],
        *,
        min_box_area: float = 0.001,
        max_box_area: float = 0.90,
        overlap_iou_threshold: float = 0.95,
    ):
        self.annotations = annotations
        self.class_names = class_names
        self._known_names: set[str] = set(class_names.values())
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.overlap_iou_threshold = overlap_iou_threshold

    def run(self) -> QualityReport:
        """Execute all checks and return a QualityReport."""
        report = QualityReport()
        report.total_frames = len(self.annotations)

        for _idx, ann in sorted(self.annotations.items()):
            if not ann.is_annotated:
                continue
            report.annotated_frames += 1

            # Frame-level: annotated flag set but nothing inside
            self._check_empty_annotation(ann, report)

            # Bounding boxes
            for bi, box in enumerate(ann.boxes):
                report.total_boxes += 1
                issues_before = len(report.issues)

                self._check_missing_label(ann.frame_index, "box", bi, box.class_name, report)
                self._check_unknown_class(ann.frame_index, "box", bi, box.class_name, report)
                self._check_invalid_bbox(ann.frame_index, bi, box, report)
                self._check_bbox_outside(ann.frame_index, bi, box, report)
                self._check_bbox_size(ann.frame_index, bi, box, report)

                if len(report.issues) == issues_before:
                    report.valid_boxes += 1

            # Duplicate / overlapping boxes
            self._check_duplicate_boxes(ann, report)

            # Polygons
            for pi, poly in enumerate(ann.polygons):
                report.total_polygons += 1
                issues_before = len(report.issues)

                self._check_missing_label(ann.frame_index, "polygon", pi, poly.class_name, report)
                self._check_unknown_class(ann.frame_index, "polygon", pi, poly.class_name, report)
                self._check_invalid_polygon(ann.frame_index, pi, poly, report)
                self._check_self_intersecting_polygon(ann.frame_index, pi, poly, report)

                if len(report.issues) == issues_before:
                    report.valid_polygons += 1

            # Duplicate / overlapping polygons
            self._check_duplicate_polygons(ann, report)

        log.info(
            f"Quality check complete — {report.annotated_frames} annotated frame(s), "
            f"{report.error_count} error(s), {report.warning_count} warning(s)"
        )
        return report

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_empty_annotation(self, ann: FrameAnnotation, report: QualityReport):
        if ann.is_annotated and not ann.boxes and not ann.polygons and not ann.classifications:
            report.issues.append(QualityIssue(
                frame_index=ann.frame_index,
                severity=Severity.WARNING,
                message="Frame marked as annotated but contains no boxes, polygons, or classifications.",
                annotation_type="frame",
            ))

    def _check_missing_label(self, frame_index: int, ann_type: str, idx: int,
                             class_name: str, report: QualityReport):
        if not class_name or not class_name.strip():
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.ERROR,
                message=f"{ann_type.capitalize()} [{idx}] has an empty/missing class label.",
                annotation_type=ann_type,
                annotation_index=idx,
            ))

    def _check_unknown_class(self, frame_index: int, ann_type: str, idx: int,
                             class_name: str, report: QualityReport):
        if class_name and class_name.strip() and self._known_names and class_name not in self._known_names:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.ERROR,
                message=f"{ann_type.capitalize()} [{idx}] class '{class_name}' is not in the known class list.",
                annotation_type=ann_type,
                annotation_index=idx,
            ))

    def _check_invalid_bbox(self, frame_index: int, idx: int, box, report: QualityReport):
        if box.width <= 0 or box.height <= 0:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.ERROR,
                message=(
                    f"Box [{idx}] '{box.class_name}' has invalid dimensions: "
                    f"w={box.width:.4f}, h={box.height:.4f}."
                ),
                annotation_type="box",
                annotation_index=idx,
            ))

    def _check_bbox_outside(self, frame_index: int, idx: int, box, report: QualityReport):
        x1 = box.x_center - box.width / 2
        y1 = box.y_center - box.height / 2
        x2 = box.x_center + box.width / 2
        y2 = box.y_center + box.height / 2
        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.WARNING,
                message=(
                    f"Box [{idx}] '{box.class_name}' extends outside image boundaries "
                    f"(x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f})."
                ),
                annotation_type="box",
                annotation_index=idx,
            ))

    def _check_bbox_size(self, frame_index: int, idx: int, box, report: QualityReport):
        area = box.width * box.height
        if area < self.min_box_area:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.WARNING,
                message=(
                    f"Box [{idx}] '{box.class_name}' is very small "
                    f"(area={area:.6f}, threshold={self.min_box_area})."
                ),
                annotation_type="box",
                annotation_index=idx,
            ))
        elif area > self.max_box_area:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.WARNING,
                message=(
                    f"Box [{idx}] '{box.class_name}' is very large "
                    f"(area={area:.4f}, threshold={self.max_box_area})."
                ),
                annotation_type="box",
                annotation_index=idx,
            ))

    def _check_duplicate_boxes(self, ann: FrameAnnotation, report: QualityReport):
        boxes = ann.boxes
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._box_iou(boxes[i], boxes[j])
                if iou > self.overlap_iou_threshold:
                    report.issues.append(QualityIssue(
                        frame_index=ann.frame_index,
                        severity=Severity.WARNING,
                        message=(
                            f"Boxes [{i}] '{boxes[i].class_name}' and [{j}] '{boxes[j].class_name}' "
                            f"overlap heavily (IoU={iou:.3f})."
                        ),
                        annotation_type="box",
                        annotation_index=i,
                    ))

    def _check_invalid_polygon(self, frame_index: int, idx: int, poly, report: QualityReport):
        if len(poly.points) < 3:
            report.issues.append(QualityIssue(
                frame_index=frame_index,
                severity=Severity.ERROR,
                message=(
                    f"Polygon [{idx}] '{poly.class_name}' has fewer than 3 points "
                    f"({len(poly.points)} point(s))."
                ),
                annotation_type="polygon",
                annotation_index=idx,
            ))

    def _check_self_intersecting_polygon(self, frame_index: int, idx: int, poly, report: QualityReport):
        pts = poly.points
        n = len(pts)
        if n < 4:
            return  # triangle can't self-intersect

        edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # adjacent edges (wrap-around)
                if self._segments_intersect(edges[i][0], edges[i][1],
                                            edges[j][0], edges[j][1]):
                    report.issues.append(QualityIssue(
                        frame_index=frame_index,
                        severity=Severity.WARNING,
                        message=(
                            f"Polygon [{idx}] '{poly.class_name}' has self-intersecting edges "
                            f"(edge {i}↔{j})."
                        ),
                        annotation_type="polygon",
                        annotation_index=idx,
                    ))
                    return  # one report per polygon is enough

    def _check_duplicate_polygons(self, ann: FrameAnnotation, report: QualityReport):
        """Flag polygons whose bounding-box IoU is extremely high."""
        polys = ann.polygons
        n = len(polys)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._poly_bbox_iou(polys[i], polys[j])
                if iou > self.overlap_iou_threshold:
                    report.issues.append(QualityIssue(
                        frame_index=ann.frame_index,
                        severity=Severity.WARNING,
                        message=(
                            f"Polygons [{i}] '{polys[i].class_name}' and [{j}] '{polys[j].class_name}' "
                            f"overlap heavily (bbox IoU={iou:.3f})."
                        ),
                        annotation_type="polygon",
                        annotation_index=i,
                    ))

    # ── Geometry helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _box_iou(a, b) -> float:
        """Compute IoU between two BoundingBox objects (normalised coords)."""
        ax1 = a.x_center - a.width / 2
        ay1 = a.y_center - a.height / 2
        ax2 = a.x_center + a.width / 2
        ay2 = a.y_center + a.height / 2

        bx1 = b.x_center - b.width / 2
        by1 = b.y_center - b.height / 2
        bx2 = b.x_center + b.width / 2
        by2 = b.y_center + b.height / 2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _poly_bbox_iou(a, b) -> float:
        """Approximate IoU using axis-aligned bounding boxes of two polygons."""
        if not a.points or not b.points:
            return 0.0

        ax = [p[0] for p in a.points]
        ay = [p[1] for p in a.points]
        bx = [p[0] for p in b.points]
        by = [p[1] for p in b.points]

        ax1, ay1, ax2, ay2 = min(ax), min(ay), max(ax), max(ay)
        bx1, by1, bx2, by2 = min(bx), min(by), max(bx), max(by)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    @staticmethod
    def _on_segment(p, q, r) -> bool:
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    @classmethod
    def _segments_intersect(cls, p1, q1, p2, q2) -> bool:
        """Check if line segment p1-q1 intersects p2-q2 (proper intersection only)."""
        d1 = cls._cross(p2, q2, p1)
        d2 = cls._cross(p2, q2, q1)
        d3 = cls._cross(p1, q1, p2)
        d4 = cls._cross(p1, q1, q2)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        if d1 == 0 and cls._on_segment(p2, p1, q2):
            return True
        if d2 == 0 and cls._on_segment(p2, q1, q2):
            return True
        if d3 == 0 and cls._on_segment(p1, p2, q1):
            return True
        if d4 == 0 and cls._on_segment(p1, q2, q1):
            return True

        return False
