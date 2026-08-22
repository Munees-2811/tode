"""Root application frame — supports video, image, and image-folder sources."""
import os
import threading
import tkinter as tk
from tkinter import messagebox

from core.annotation_manager import AnnotationManager
from core.exporter import DatasetExporter
from core.frame_extractor import FrameExtractor
from core.image_frame_extractor import ImageFrameExtractor
from core.image_loader import ImageLoader
from core.video_loader import VideoLoader
from core.yolo_annotator import YOLOAnnotator
from models.annotation_model import BoundingBox
from storage.frame_storage import FrameStorage
from storage.label_storage import LabelStorage
from ui.annotation_panel import AnnotationPanel
from ui.export_dialog import ExportDialog
from ui.log_viewer import LogViewer
from ui.segmentation_panel import SegmentationPanel
from ui.source_dialog import SourceDialog
from ui.video_player import VideoPlayer
from utils.config import ACCENT, BG_DARK, BG_PANEL
from utils.logger import get_logger

log = get_logger("ui.MainWindow")

_SOURCE_LABELS = {
    "video":        "🎬 Video",
    "image":        "🖼 Image",
    "image_folder": "📂 Image Folder",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _find_images_recursive(folder: str):
    """
    Scan folder AND all subfolders for supported image files.
    Returns sorted list of absolute paths.
    """
    found = []
    for root, dirs, files in os.walk(folder):
        # Sort dirs so traversal is deterministic
        dirs.sort()
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                found.append(os.path.join(root, f))
    return found


class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_DARK)
        self.manager: AnnotationManager | None = None
        self._busy        = False
        self._log_viewer  = None
        self._source_type = None
        self._build_ui()
        log.info("MainWindow initialised")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()

        content = tk.Frame(self, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True)

        self.player = VideoPlayer(
            content,
            on_frame_change=self._on_frame_change,
            on_box_drawn=self._on_box_drawn,
            on_open_request=self._open_source,
            on_box_edited=self._on_box_edited,
            on_box_selected=self._on_box_selected_in_canvas,
            on_polygon_drawn=self._on_polygon_drawn,
            on_mode_change=self._on_mode_change,
        )
        self.player.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                         padx=6, pady=6)

        self.ann_panel = AnnotationPanel(
            content,
            on_yolo_click              = self._run_yolo,
            on_yolo_all_click          = self._run_yolo_all,
            on_save_click              = self._save,
            on_clear_click             = self._clear_frame,
            on_delete_box              = self._delete_box,
            on_conf_change             = self._on_conf_change,
            on_model_change            = self._on_model_change,
            on_box_select              = self._on_box_selected_in_list,
            on_accept_suggestion      = self._accept_suggestion,
            on_accept_all_suggestions = self._accept_all_suggestions,
            on_reject_all_suggestions = self._reject_all_suggestions,
        )
        self.ann_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

        # ── semantic segmentation panel (hidden until polygon mode) ────────
        self._class_color_map: dict[str, str] = {}
        self.seg_panel = SegmentationPanel(
            content,
            on_save_click                  = self._save,
            on_clear_click                 = self._clear_seg_frame,
            on_delete_poly                 = self._delete_polygon,
            on_poly_select                 = self._on_poly_selected,
            on_class_changed               = self._on_seg_class_changed,
            on_opacity_change              = self._on_seg_opacity_changed,
            on_auto_seg_click              = self._run_yolo_seg,
            on_auto_seg_all_click          = self._run_yolo_seg_all,
            on_accept_poly_suggestion      = lambda idx: self._accept_suggestion(idx, is_polygon=True),
            on_accept_all_poly_suggestions = lambda: self._accept_all_suggestions(is_polygon=True),
            on_reject_all_poly_suggestions = lambda: self._reject_all_suggestions(is_polygon=True),
        )
        # seg_panel is shown/hidden by the polygon mode button in video_player
        self.seg_panel.pack_forget()

        self._build_status()
        self._build_progress()
        self._bind_shortcuts()

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=46)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        def btn(parent, text, cmd, bg=ACCENT, hover="#9d8fff"):
            b = tk.Button(
                parent, text=text, command=cmd,
                bg=bg, fg="white", relief=tk.FLAT,
                padx=11, pady=7,
                font=("Consolas", 9, "bold"),
                cursor="hand2",
                activebackground=hover,
                activeforeground="white",
                bd=0,
            )
            b.pack(side=tk.LEFT, padx=2, pady=6)
            b.bind("<Enter>", lambda _e, b=b, h=hover: b.config(bg=h))
            b.bind("<Leave>", lambda _e, b=b, c=bg:   b.config(bg=c))
            return b

        def sep():
            tk.Frame(bar, bg="#3a3a5e", width=1).pack(
                side=tk.LEFT, fill=tk.Y, pady=10, padx=3)

        btn(bar, "📂  Open",   self._open_source,    bg="#4a3a8a", hover="#7a6adf")
        sep()
        btn(bar, "💾  Save",   self._save,            bg="#2d7a4e", hover="#3da060")
        btn(bar, "📤  Export", self._export_dataset,  bg="#1f7a8c", hover="#2a9aae")
        sep()
        btn(bar, "📋  Logs",   self._show_logs,       bg="#3a4a6a", hover="#4a5a7a")

        self._badge_icon = tk.Label(
            bar, text="  ●",
            bg=BG_PANEL, fg="#444466",
            font=("Consolas", 10),
        )
        self._badge_icon.pack(side=tk.RIGHT, padx=(0, 2))

        self._badge_text = tk.Label(
            bar, text="No source loaded  ",
            bg=BG_PANEL, fg="#555577",
            font=("Consolas", 8, "italic"),
        )
        self._badge_text.pack(side=tk.RIGHT)

    def _build_status(self):
        bar = tk.Frame(self, bg="#1a1a2e", height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self._mode_var = tk.StringVar(value="VIEW")
        self._mode_chip = tk.Label(
            bar, textvariable=self._mode_var,
            bg=ACCENT, fg="white",
            font=("Consolas", 8, "bold"),
            padx=6, pady=2,
        )
        self._mode_chip.pack(side=tk.LEFT, padx=(8, 6), pady=4)

        tk.Frame(bar, bg="#3a3a5e", width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=4)

        self.status_var = tk.StringVar(
            value="No source loaded. Click 📂 Open to begin."
        )
        tk.Label(
            bar, textvariable=self.status_var,
            bg="#1a1a2e", fg="#aaaacc",
            font=("Consolas", 8),
        ).pack(side=tk.LEFT, padx=10)

        self._ann_count_var = tk.StringVar(value="")
        tk.Label(
            bar, textvariable=self._ann_count_var,
            bg="#1a1a2e", fg="#55cc77",
            font=("Consolas", 8),
        ).pack(side=tk.RIGHT, padx=12)

    def _build_progress(self):
        from tkinter import ttk
        pf = tk.Frame(self, bg=BG_PANEL, height=4)
        pf.pack(fill=tk.X, side=tk.BOTTOM)
        pf.pack_propagate(False)
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "App.Horizontal.TProgressbar",
            troughcolor=BG_PANEL, background=ACCENT, thickness=4,
        )
        self._progress = ttk.Progressbar(
            pf, mode="indeterminate",
            style="App.Horizontal.TProgressbar",
        )
        self._progress_visible = False

    # ── keyboard shortcuts (labelImg-style) ───────────────────────────────────
    def _bind_shortcuts(self):
        root = self.master
        mappings = {
            "<Left>":      lambda _e: self._nav(-1),
            "a":           lambda _e: self._nav(-1),
            "<Right>":     lambda _e: self._nav(+1),
            "d":           lambda _e: self._nav(+1),
            "<Home>":      lambda _e: self._nav("first"),
            "<End>":       lambda _e: self._nav("last"),
            "w":           lambda _e: self.player.set_draw_mode(),
            "v":           lambda _e: self.player.set_view_mode(),
            "<Escape>":    lambda _e: self.player.set_view_mode(),
            "<Control-s>": lambda _e: self._save(),
            "<Control-e>": lambda _e: self._export_dataset(),
            "<Control-o>": lambda _e: self._open_source(),
            "<Delete>":    lambda _e: self._clear_frame(),
            "y":           lambda _e: self._handle_y_key(),
        }
        for key, fn in mappings.items():
            root.bind(key, fn)
        log.debug("Keyboard shortcuts bound")

    def _handle_y_key(self):
        if self.player.is_polygon_mode() or self.player.is_magic_wand_mode():
            self._run_yolo_seg()
        else:
            self._run_yolo()

    def _nav(self, where):
        """Move slider — accepts -1, +1, 'first', 'last'."""
        if where == "first":
            self.player._go_first()
        elif where == "last":
            self.player._go_last()
        elif where == -1:
            self.player._prev()
        elif where == +1:
            self.player._next()

    # ── progress ──────────────────────────────────────────────────────────────
    def _show_progress(self):
        if not self._progress_visible:
            self._progress.pack(fill=tk.X)
            self._progress.start(12)
            self._progress_visible = True

    def _hide_progress(self):
        if self._progress_visible:
            self._progress.stop()
            self._progress.pack_forget()
            self._progress_visible = False

    def _run_in_thread(self, task_fn, done_fn=None, error_fn=None):
        self._busy = True
        self._show_progress()

        def _safe_done(res):
            try:
                if done_fn:
                    done_fn(res)
            except Exception as exc:
                log.error(f"UI update callback error: {exc}", exc_info=True)
                messagebox.showerror("UI Error", f"Failed to update UI:\n{exc}")

        def _worker():
            try:
                result = task_fn()
                self.after(0, lambda r=result: _safe_done(r))
            except Exception as exc:
                # FIX: Python 3.13 lambda scoping bug — capture exc explicitly
                _exc = exc
                log.error(f"Background task error: {_exc}", exc_info=True)
                if error_fn:
                    self.after(0, lambda e=_exc: error_fn(e))
                else:
                    self.after(
                        0, lambda e=_exc: messagebox.showerror("Error", str(e))
                    )
            finally:
                self.after(0, self._task_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _task_done(self):
        self._busy = False
        self._hide_progress()

    # ── open source actions ───────────────────────────────────────────────────
    def _open_source(self):
        """Unified tabbed source dialog."""
        if self._busy:
            return
        dlg = SourceDialog(self.master)
        if dlg.result:
            self._load_from_result(dlg.result)

    # ── unified loader dispatcher ─────────────────────────────────────────────
    def _load_from_result(self, result: dict):
        src_type = result["type"]
        path     = result["path"]
        self._source_type = src_type
        self._update_badge(src_type)
        log.info(f"Loading source — type={src_type}, path={path}")

        if src_type == "video":
            self._load_video(path, step=result.get("step", 1))
        elif src_type in ("image", "image_folder"):
            self._load_images(path)

    # ── video loader ──────────────────────────────────────────────────────────
    def _load_video(self, path: str, step: int = 1):
        self._set_status(f"Opening video: {os.path.basename(path)}…")

        def _on_bg_progress(done, total):
            pct = (100 * done) // max(1, total)
            self.after(0, lambda d=done, t=total, p=pct: self._set_status(
                f"Extracting frames in background… {d}/{t} ({p}%)"
                if d < t else f"All {t} frames extracted."
            ))

        def _work():
            loader    = VideoLoader(path)
            loader.open()
            extractor = FrameExtractor(loader, step=step, save_frames=True)
            yolo      = YOLOAnnotator()
            vname = os.path.splitext(os.path.basename(path))[0]
            mgr   = AnnotationManager(
                loader, extractor, yolo,
                FrameStorage(vname),
                LabelStorage(vname),
            )
            mgr.load_video(on_progress=_on_bg_progress)
            mgr.load_existing_labels()
            return mgr

        def _done(mgr: AnnotationManager):
            self.manager = mgr
            self.player.load(
                mgr.loader, mgr.all_frame_indices(),
                frame_path_provider=self._frame_path_for,
            )
            self._set_status(
                f"Video ready — '{os.path.basename(path)}'  "
                f"| {mgr.loader.total_frames} frames "
                f"| {mgr.loader.fps:.0f} fps  "
                f"(frames extract in background — start annotating now)"
            )
            self._refresh_ann_count()

        def _err(exc):
            messagebox.showerror("Video Load Error", str(exc))
            self._set_status("Failed to load video.")

        self._run_in_thread(_work, _done, _err)

    # ── image loader ──────────────────────────────────────────────────────────
    def _load_images(self, path: str):
        is_folder = os.path.isdir(path)
        label     = "folder" if is_folder else "image"
        self._set_status(f"Loading {label}: {os.path.basename(path)}…")
        log.info(f"Loading images — is_folder={is_folder}, path={path}")

        def _work():
            # ── FIX: if folder has no images at root, scan subfolders ─────────
            if is_folder:
                all_images = _find_images_recursive(path)
                if not all_images:
                    raise FileNotFoundError(
                        f"No images found in folder or subfolders:\n{path}\n\n"
                        f"Supported formats: JPG, PNG, BMP, TIFF, WEBP\n\n"
                        f"Make sure your images are inside the selected folder."
                    )
                log.info(f"Found {len(all_images)} image(s) in: {path}")

            loader = ImageLoader(path)
            loader.open()

            if loader.total_frames == 0:
                raise ValueError(
                    "No supported images found.\n"
                    "Supported formats: JPG, PNG, BMP, TIFF, WEBP"
                )

            extractor = ImageFrameExtractor(loader, copy_files=True)
            yolo      = YOLOAnnotator()

            src_name = (
                os.path.basename(path.rstrip("/\\")) or "images"
            )
            # Sanitise name for use as directory
            src_name = "".join(
                c if c.isalnum() or c in "-_." else "_"
                for c in src_name
            )

            mgr = AnnotationManager(
                loader, extractor, yolo,
                FrameStorage(src_name),
                LabelStorage(src_name),
            )
            self.after(0, lambda: self._set_status(
                f"Processing {loader.total_frames} image(s)…"
            ))
            mgr.load_video()           # duck-typed — works for images too
            mgr.load_existing_labels()
            return mgr

        def _done(mgr: AnnotationManager):
            self.manager = mgr
            indices = mgr.all_frame_indices()
            self.player.load(
                mgr.loader, indices,
                frame_path_provider=self._frame_path_for,
            )
            noun = "images" if is_folder else "image"
            self._set_status(
                f"{'Folder' if is_folder else 'Image'} loaded — "
                f"'{os.path.basename(path)}'  |  {len(indices)} {noun}"
            )
            log.info(f"Image source ready — {len(indices)} item(s)")
            self._refresh_ann_count()

        def _err(exc):
            log.error(f"Image load error: {exc}", exc_info=True)
            messagebox.showerror(
                "Image Load Error",
                f"Could not load:\n{path}\n\n{exc}",
            )
            self._set_status("Failed to load image source.")

        self._run_in_thread(_work, _done, _err)

    # ── frame change callback ─────────────────────────────────────────────────
    # ── frame change & overlay refresh ────────────────────────────────────────
    def _refresh_current_frame_overlays(self):
        if self.manager is None:
            return
        idx = self.player.current_frame_index
        ann = self.manager.get_annotation(idx)
        boxes = ann.boxes if ann else []
        suggs = ann.suggested_boxes if ann else []
        polygons = ann.polygons if ann else []
        sugg_polys = ann.suggested_polygons if ann else []

        self.player.set_overlay_boxes(boxes)
        self.player.set_overlay_suggested_boxes(suggs)
        self.player.set_overlay_polygons(polygons)
        self.player.set_overlay_suggested_polygons(sugg_polys)

        self.ann_panel.update_boxes(boxes, self.manager.yolo.class_names, suggested_boxes=suggs)
        class_names = list(self.manager.yolo.class_names.values())
        self.seg_panel.update_polygons(polygons, class_names)
        self._sync_color_map()
        self._refresh_ann_count()

    def _on_frame_change(self, frame_index: int, bgr_frame):
        self._refresh_current_frame_overlays()

    # ── AI suggestion verification callbacks ──────────────────────────────────
    def _accept_suggestion(self, sugg_index: int, is_polygon: bool = False):
        if self.manager is None:
            return
        idx = self.player.current_frame_index
        accepted = self.manager.accept_suggestion(idx, sugg_index, is_polygon=is_polygon)
        if accepted:
            self._refresh_current_frame_overlays()
            self._set_status(f"Accepted AI suggestion [{sugg_index}] — '{accepted.class_name}'.")

    def _accept_all_suggestions(self, is_polygon: bool = False):
        if self.manager is None:
            return
        idx = self.player.current_frame_index
        conf = self.ann_panel.get_confidence_threshold()
        count = self.manager.accept_all_suggestions(frame_index=idx, min_confidence=conf, is_polygon=is_polygon)
        self._refresh_current_frame_overlays()
        self._set_status(f"Accepted {count} AI suggestion(s) on index {idx + 1}.")

    def _reject_all_suggestions(self, is_polygon: bool = False):
        if self.manager is None:
            return
        idx = self.player.current_frame_index
        self.manager.reject_all_suggestions(frame_index=idx, is_polygon=is_polygon)
        self._refresh_current_frame_overlays()
        self._set_status(f"Rejected all AI suggestions on index {idx + 1}.")

    # ── mode change callback (panel swap) ─────────────────────────────────────
    def _on_mode_change(self, mode: str) -> None:
        self._mode_var.set(mode.upper())
        if mode == "polygon":
            self._mode_chip.config(bg="#2a9d5c")
        elif mode == "draw":
            self._mode_chip.config(bg="#e05c5c")
        else:
            self._mode_chip.config(bg=ACCENT)

        if mode == "polygon":
            self.ann_panel.pack_forget()
            self.seg_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)
            if self.manager:
                self.seg_panel.set_class_names(
                    list(self.manager.yolo.class_names.values())
                )
        else:
            self.seg_panel.pack_forget()
            self.ann_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

    # ── polygon drawn callback ────────────────────────────────────────────────
    def _on_polygon_drawn(self, points: list):
        if self.manager is None:
            return
        from models.annotation_model import PolygonAnnotation
        idx      = self.player.current_frame_index
        cls_name = self.seg_panel.get_selected_class()
        class_names = self.manager.yolo.class_names
        cls_id   = next(
            (k for k, v in class_names.items()
             if v.lower() == cls_name.lower()), 0
        )
        poly = PolygonAnnotation(
            class_id=cls_id, class_name=cls_name, points=points
        )
        self.manager.add_polygon(idx, poly)
        self._refresh_current_frame_overlays()
        self._set_status(
            f"Polygon added to frame {idx + 1} — '{cls_name}', {len(points)} pts."
        )

    # ── seg panel callbacks ───────────────────────────────────────────────────

    def _on_seg_class_changed(self, class_name: str, color: str) -> None:
        """Called when user picks a different semantic class in seg_panel."""
        self._class_color_map[class_name] = color
        self.player.set_active_poly_color(color)
        self.player.set_class_color_map(dict(self._class_color_map))

    def _on_seg_opacity_changed(self, value: float) -> None:
        self.player.set_poly_opacity(value)

    def _on_poly_selected(self, index: int | None) -> None:
        """Highlight the selected polygon in the canvas (future use)."""
        pass

    def _delete_polygon(self, poly_index: int, is_suggestion: bool = False) -> None:
        if not self._require_manager():
            return
        idx = self.player.current_frame_index
        if is_suggestion:
            self.manager.reject_suggestion(idx, poly_index, is_polygon=True)
            self._set_status(f"Rejected AI polygon suggestion [{poly_index}] from frame {idx + 1}.")
        else:
            self.manager.remove_polygon(idx, poly_index)
            self._set_status(f"Deleted polygon [{poly_index}] from frame {idx + 1}.")
        self._refresh_current_frame_overlays()

    def _clear_seg_frame(self) -> None:
        if not self._require_manager():
            return
        idx = self.player.current_frame_index
        ann = self.manager.get_annotation(idx)
        if ann and ann.polygons:
            if not messagebox.askyesno(
                "Clear polygons",
                f"Remove all {len(ann.polygons)} polygon(s) on this frame?",
            ):
                return
        self.manager.clear_polygons(idx)
        self._refresh_current_frame_overlays()
        self._set_status(f"Cleared all polygons on frame {idx + 1}.")

    def _sync_color_map(self) -> None:
        """Push current class→colour mapping into the canvas."""
        for cls in self.seg_panel._classes:
            self._class_color_map[cls["name"]] = cls["color"]
        self.player.set_class_color_map(dict(self._class_color_map))

    # ── manual box drawn ──────────────────────────────────────────────────────
    def _on_box_drawn(self, x1_n: float, y1_n: float,
                      x2_n: float, y2_n: float):
        if self.manager is None:
            return

        cls_name    = self.ann_panel.get_selected_class()
        class_names = self.manager.yolo.class_names
        cls_id      = next(
            (k for k, v in class_names.items()
             if v.lower() == cls_name.lower()), 0
        )

        box = BoundingBox(
            class_id   = cls_id,
            class_name = cls_name,
            x_center   = (x1_n + x2_n) / 2,
            y_center   = (y1_n + y2_n) / 2,
            width      = x2_n - x1_n,
            height     = y2_n - y1_n,
            confidence = 1.0,
        )

        idx = self.player.current_frame_index
        self.manager.add_box(idx, box)
        self._refresh_current_frame_overlays()
        src_label = (
            "image" if self._source_type in ("image", "image_folder")
            else "frame"
        )
        self._set_status(
            f"Manual box added — '{cls_name}' on {src_label} {idx + 1}."
        )

    # ── box edit / select callbacks ───────────────────────────────────────────
    def _on_box_edited(self, box_index: int,
                       x1_n: float, y1_n: float,
                       x2_n: float, y2_n: float,
                       is_suggestion: bool = False):
        if self.manager is None:
            return
        idx = self.player.current_frame_index
        ann = self.manager.get_annotation(idx)
        if not ann:
            return

        if is_suggestion:
            if 0 <= box_index < len(ann.suggested_boxes):
                box = ann.suggested_boxes[box_index]
                box.x_center = (x1_n + x2_n) / 2
                box.y_center = (y1_n + y2_n) / 2
                box.width    = x2_n - x1_n
                box.height   = y2_n - y1_n
                box.confidence = 1.0
                ann.accept_suggested_box(box_index)
                self._set_status(f"Edited & accepted suggestion [{box_index}] → '{box.class_name}'")
        else:
            if 0 <= box_index < len(ann.boxes):
                box = ann.boxes[box_index]
                box.x_center = (x1_n + x2_n) / 2
                box.y_center = (y1_n + y2_n) / 2
                box.width    = x2_n - x1_n
                box.height   = y2_n - y1_n
                ann.is_annotated = bool(ann.boxes)
                self._set_status(f"Edited box [{box_index}] — '{box.class_name}'")
        self._refresh_current_frame_overlays()

    def _on_box_selected_in_canvas(self, box_index, is_suggestion: bool = False):
        """Sync canvas → listbox highlight."""
        self.ann_panel.set_selected_box(box_index, is_suggestion=is_suggestion)

    def _on_box_selected_in_list(self, box_index, is_suggestion: bool = False):
        """Sync listbox → canvas highlight."""
        self.player.set_selected_box(box_index, is_suggestion=is_suggestion)

    # ── delete selected box ───────────────────────────────────────────────────
    def _delete_box(self, box_index: int, is_suggestion: bool = False):
        if not self._require_manager():
            return
        idx = self.player.current_frame_index
        if is_suggestion:
            self.manager.reject_suggestion(idx, box_index)
            self._set_status(f"Rejected AI suggestion [{box_index}] from index {idx + 1}.")
        else:
            self.manager.remove_box(idx, box_index)
            self._set_status(f"Deleted box [{box_index}] from index {idx + 1}.")
        self._refresh_current_frame_overlays()

    # ── confidence change ─────────────────────────────────────────────────────
    def _on_conf_change(self, val: float):
        if self.manager:
            self.manager.yolo.confidence = val
            log.debug(f"Confidence updated → {val:.2f}")
            self._refresh_current_frame_overlays()

    # ── model change ──────────────────────────────────────────────────────────
    def _on_model_change(self, model: str):
        yolo = self.manager.yolo if self.manager else None
        target = yolo or __import__("core.yolo_annotator", fromlist=["YOLOAnnotator"]).YOLOAnnotator()
        self._set_status(f"Loading model '{model}'…")
        log.info(f"Model change requested → {model}")

        def _work():
            if self.manager:
                self.manager.yolo.reload(model)
            else:
                target.reload(model)

        def _done(_):
            name = model.split("/")[-1].split("\\")[-1]
            self._set_status(f"Model ready: {name}")

        def _err(exc):
            messagebox.showerror("Model Load Error",
                                 f"Could not load '{model}':\n{exc}")
            self._set_status("Model load failed.")

        self._run_in_thread(_work, _done, _err)

    # ── YOLO single frame ─────────────────────────────────────────────────────
    def _run_yolo(self):
        if not self._require_manager() or self._busy:
            return
        idx        = self.player.current_frame_index
        conf       = self.ann_panel.get_confidence_threshold()
        cls_filter = self.ann_panel.get_class_filter()
        self.manager.yolo.confidence = conf
        self._set_status(f"Running YOLO on index {idx + 1}…")
        log.info(f"YOLO single — idx={idx}, conf={conf}, filter={cls_filter}")

        def _work():
            ann = self.manager.auto_annotate_frame(idx)
            if cls_filter:
                ann.suggested_boxes = [
                    b for b in ann.suggested_boxes
                    if b.class_name.lower() in cls_filter
                ]
            return ann

        def _done(ann):
            self._refresh_current_frame_overlays()
            self._set_status(
                f"YOLO: {len(ann.suggested_boxes)} AI suggestion(s) at index {idx + 1}. Review and verify."
            )
            self._refresh_ann_count()

        self._run_in_thread(_work, _done)

    # ── YOLO all frames ───────────────────────────────────────────────────────
    def _run_yolo_all(self):
        if not self._require_manager() or self._busy:
            return
        conf       = self.ann_panel.get_confidence_threshold()
        cls_filter = self.ann_panel.get_class_filter()
        self.manager.yolo.confidence = conf

        self._set_status("Running YOLO on all frames…")
        log.info(f"YOLO all — conf={conf}, filter={cls_filter}")

        def _progress(done, tot):
            self.after(0, lambda d=done, t=tot: self._set_status(
                f"YOLO annotating… {d}/{t}"
            ))

        def _work():
            self.manager.auto_annotate_all(progress_callback=_progress)
            if cls_filter:
                for ann in self.manager._annotations.values():
                    ann.suggested_boxes = [
                        b for b in ann.suggested_boxes
                        if b.class_name.lower() in cls_filter
                    ]
            return self.manager.total_count

        def _done(count):
            self._refresh_current_frame_overlays()
            self._set_status("YOLO AI suggestions generated across all frames. Review & verify.")
            self._refresh_ann_count()

        self._run_in_thread(_work, _done)

    # ── YOLO polygon single frame ─────────────────────────────────────────────
    def _run_yolo_seg(self):
        if not self._require_manager() or self._busy:
            return
        idx = self.player.current_frame_index
        conf = self.ann_panel.get_confidence_threshold()
        cls_filter = self.ann_panel.get_class_filter()
        selected_model = self.seg_panel.get_model_name()
        if selected_model and self.manager.yolo.model_path != selected_model:
            log.info(f"Switching segmentation model to: {selected_model}")
            self.manager.yolo.reload(selected_model)
        self.manager.yolo.confidence = conf
        self._set_status(f"Running YOLO auto polygon segmentation on index {idx + 1}…")
        log.info(f"YOLO seg single — idx={idx}, model={selected_model}, conf={conf}, filter={cls_filter}")

        def _work():
            ann = self.manager.auto_annotate_polygons_frame(idx)
            if cls_filter:
                ann.suggested_polygons = [
                    p for p in ann.suggested_polygons
                    if p.class_name.lower() in cls_filter
                ]
            return ann

        def _done(ann):
            self._refresh_current_frame_overlays()
            self._set_status(
                f"YOLO Seg: {len(ann.suggested_polygons)} polygon suggestion(s) at index {idx + 1}."
            )
            self._refresh_ann_count()

        self._run_in_thread(_work, _done)

    # ── YOLO polygon all frames ───────────────────────────────────────────────
    def _run_yolo_seg_all(self):
        if not self._require_manager() or self._busy:
            return
        conf  = self.ann_panel.get_confidence_threshold()
        cls_filter = self.ann_panel.get_class_filter()
        selected_model = self.seg_panel.get_model_name()
        if selected_model and self.manager.yolo.model_path != selected_model:
            log.info(f"Switching segmentation model to: {selected_model}")
            self.manager.yolo.reload(selected_model)
        self.manager.yolo.confidence = conf

        self._set_status("Running YOLO polygon segmentation on all frames…")
        log.info(f"YOLO seg all — model={selected_model}, conf={conf}, filter={cls_filter}")

        def _progress(done, tot):
            self.after(0, lambda d=done, t=tot: self._set_status(
                f"YOLO polygon annotating… {d}/{t}"
            ))

        def _work():
            self.manager.auto_annotate_polygons_all(progress_callback=_progress)
            if cls_filter:
                for ann in self.manager._annotations.values():
                    ann.suggested_polygons = [
                        p for p in ann.suggested_polygons
                        if p.class_name.lower() in cls_filter
                    ]
            return self.manager.total_count

        def _done(count):
            self._refresh_current_frame_overlays()
            self._set_status("YOLO polygon AI suggestions complete. Review and verify.")
            self._refresh_ann_count()

        self._run_in_thread(_work, _done)

        self._run_in_thread(_work, _done)

    # ── save annotations ──────────────────────────────────────────────────────
    def _save(self):
        if not self._require_manager() or self._busy:
            return
        self._set_status("Saving annotations…")

        def _work():
            self.manager.save_annotations()
            return self.manager.annotated_count

        def _done(count):
            self._set_status(f"Saved {count} annotation file(s).")
            self._refresh_ann_count()

        self._run_in_thread(_work, _done)

    # ── export dataset ────────────────────────────────────────────────────────
    def _export_dataset(self):
        if not self._require_manager() or self._busy:
            return
        if self.manager.annotated_count == 0:
            messagebox.showinfo(
                "Nothing to export",
                "No annotated frames yet. Annotate at least one frame, "
                "then export.",
            )
            return

        # Default to ~/Documents/labeled_img/<source_name>/ so exported
        # datasets land somewhere users can find easily.
        docs = os.path.join(os.path.expanduser("~"), "Documents")
        if not os.path.isdir(docs):
            docs = os.path.expanduser("~")
        src_name = self.manager.f_store.video_name or "dataset"
        default_out = os.path.join(docs, "labeled_img", src_name)
        dlg = ExportDialog(
            self.master,
            default_dir=default_out,
            total_annotated=self.manager.annotated_count,
        )
        if not dlg.result:
            return

        fmt     = dlg.result["format"]
        out_dir = dlg.result["output_dir"]
        self._set_status(f"Exporting as {fmt.upper()} → {out_dir}…")
        log.info(f"Export started — fmt={fmt}, out={out_dir}, settings={dlg.result}")

        exporter = DatasetExporter(
            annotations = self.manager._annotations,
            class_names = self.manager.yolo.class_names,
            output_dir  = out_dir,
        )

        def _progress(done, total):
            self.after(0, lambda d=done, t=total: self._set_status(
                f"Exporting dataset… {d}/{t}"
            ))

        def _work():
            return exporter.export(
                fmt=fmt,
                progress_callback=_progress,
                split=dlg.result.get("split", False),
                train_ratio=dlg.result.get("train_ratio", 0.70),
                val_ratio=dlg.result.get("val_ratio", 0.20),
                test_ratio=dlg.result.get("test_ratio", 0.10),
                seed=dlg.result.get("seed", 42),
                use_random_seed=dlg.result.get("use_random_seed", False),
            )

        def _done(summary: dict):
            if summary.get("format") == "yolo_split":
                msg = (
                    f"Format : YOLO Dataset Split\n"
                    f"Total Images : {summary['total_images']}\n"
                    f"  ├─ Train : {summary['train_images']}\n"
                    f"  ├─ Val   : {summary['val_images']}\n"
                    f"  └─ Test  : {summary['test_images']}\n"
                    f"Classes ({len(summary['classes'])}): {', '.join(summary['classes'])}\n\n"
                    f"Saved to:\n{summary['output_dir']}"
                )
                self._set_status(
                    f"Dataset Split complete — Train: {summary['train_images']}, "
                    f"Val: {summary['val_images']}, Test: {summary['test_images']} → {summary['output_dir']}"
                )
            else:
                msg = (
                    f"Format : {summary['format'].upper()}\n"
                    f"Images : {summary['images']}\n"
                    f"Labels : {summary['labels']}\n"
                    f"Classes: {', '.join(summary['classes'])}\n\n"
                    f"Saved to:\n{summary['output_dir']}"
                )
                self._set_status(
                    f"Export complete — {summary['images']} images, "
                    f"{summary['labels']} labels → {summary['output_dir']}"
                )
            messagebox.showinfo("Export Complete", msg)

        def _err(exc):
            messagebox.showerror("Export Error", str(exc))
            self._set_status("Export failed.")

        self._run_in_thread(_work, _done, _err)

    # ── clear current frame ───────────────────────────────────────────────────
    def _clear_frame(self):
        if not self._require_manager():
            return
        idx = self.player.current_frame_index
        ann = self.manager.get_annotation(idx)
        if ann and ann.boxes:
            if not messagebox.askyesno(
                "Clear frame",
                f"Remove all {len(ann.boxes)} box(es) on this frame?",
            ):
                return
        self.manager.clear_frame(idx)
        self.player.set_overlay_boxes([])
        self.ann_panel.update_boxes([], {})
        self._set_status(f"Cleared annotations for index {idx + 1}.")
        self._refresh_ann_count()

    # ── log viewer ────────────────────────────────────────────────────────────
    def _show_logs(self):
        if self._log_viewer is None or not self._log_viewer.winfo_exists():
            self._log_viewer = LogViewer(self.master)
            log.info("Log viewer opened")
        else:
            self._log_viewer.deiconify()
            self._log_viewer.lift()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _frame_path_for(self, frame_index: int) -> str:
        """Resolve the on-disk PNG path for a frame index, or '' if missing."""
        if self.manager is None:
            return ""
        ann = self.manager.get_annotation(frame_index)
        return ann.frame_path if ann else ""

    def _require_manager(self) -> bool:
        if self.manager is None:
            messagebox.showinfo("No source",
                                "Please open a source first.")
            return False
        return True

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        log.info(f"[STATUS] {msg}")
        self.update_idletasks()

    def _refresh_ann_count(self):
        if self.manager is None:
            self._ann_count_var.set("")
            return
        ann = self.manager.annotated_count
        tot = self.manager.total_count
        self._ann_count_var.set(f"✔ {ann}/{tot} annotated")

    def _update_badge(self, src_type: str):
        _SOURCE_LABELS = {
            "video":        ("🎬", "Video",        "#6a4fbf"),
            "image":        ("🖼", "Image",         "#2d7a4e"),
            "image_folder": ("📂", "Image Folder", "#1f7a8c"),
        }
        icon, label, color = _SOURCE_LABELS.get(src_type, ("●", src_type, ACCENT))
        self._badge_icon.config(text=f"  {icon}", fg=color)
        self._badge_text.config(text=f"{label}  ", fg="#ccccee")

    def setup_drag_drop(self):
        _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        self.master.drop_target_register("DND_Files")  # type: ignore[attr-defined]

        def _on_drop(event):
            raw = event.data.strip()
            # tkinterdnd2 wraps paths containing spaces in { }
            if raw.startswith("{") and raw.endswith("}"):
                path = raw[1:-1]
            else:
                path = raw.split()[0]
            ext = os.path.splitext(path)[1].lower()
            if ext in _VIDEO_EXTS:
                self._load_from_result({"type": "video", "path": path, "step": 1})
            elif ext in SUPPORTED_EXTS:
                self._load_from_result({"type": "image", "path": path, "step": 1})
            elif os.path.isdir(path):
                self._load_from_result({"type": "image_folder", "path": path, "step": 1})
            else:
                log.warning(f"Dropped file has unsupported type: {path}")

        self.master.dnd_bind("<<Drop>>", _on_drop)  # type: ignore[attr-defined]

    def on_close(self):
        log.info("Application closing — releasing resources")
        if self.manager:
            self.manager.loader.release()
        self.master.destroy()

