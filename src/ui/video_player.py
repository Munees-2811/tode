"""
Canvas widget — displays frames + overlays.
Supports two modes:
  VIEW   : navigate frames only
  DRAW   : click-drag to draw a new bounding box
"""
import os
import tkinter as tk
from collections.abc import Callable

from models.annotation_model import BoundingBox, PolygonAnnotation
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT
from utils.image_utils import bgr_to_photoimage, safe_imread


class VideoPlayer(tk.Frame):
    MODE_VIEW = "view"
    MODE_DRAW = "draw"
    MODE_POLYGON = "polygon"

    def __init__(self, master, on_frame_change: Callable = None,
                 on_box_drawn: Callable = None,
                 on_open_request: Callable = None,
                 on_box_edited: Callable = None,
                 on_box_selected: Callable = None,
                 on_polygon_drawn: Callable = None,
                 on_mode_change: Callable = None):
        """
        Parameters
        ----------
        on_frame_change  : callable(frame_index, bgr_frame)
        on_box_drawn     : callable(x1_norm, y1_norm, x2_norm, y2_norm)
        on_box_edited    : callable(box_index, x1_norm, y1_norm, x2_norm, y2_norm)
        on_box_selected  : callable(box_index_or_None)
        on_mode_change   : callable(mode: str)  — "view" | "draw" | "polygon"
        """
        super().__init__(master, bg=BG_DARK)
        self._on_change = on_frame_change
        self._on_box_drawn = on_box_drawn
        self._on_open_request = on_open_request
        self._on_box_edited = on_box_edited
        self._on_box_selected = on_box_selected
        self._on_polygon_drawn = on_polygon_drawn
        self._on_mode_change = on_mode_change
        self._frame_path_provider: Callable[[int], str] | None = None

        self._loader = None
        self._indices: list[int] = []
        self._pos = 0
        self._boxes: list[BoundingBox] = []
        self._suggested_boxes: list[BoundingBox] = []
        self._polygons: list[PolygonAnnotation] = []
        self._suggested_polygons: list[PolygonAnnotation] = []
        self._photo = None
        self._photo_size = (0, 0)
        self._image_id = None
        self._last_frame_id = None
        self._current_frame = None

        # ── draw-mode state ───────────────────────────────────────────────────
        self._mode = self.MODE_VIEW
        self._draw_start: tuple | None = None  # (canvas_x, canvas_y)
        self._draw_rect = None  # canvas rect id

        # ── edit-mode state ───────────────────────────────────────────────────
        self._selected_idx: int | None = None
        self._selected_is_suggestion: bool = False
        self._edit_handle: str | None = None
        self._edit_drag_start: tuple | None = None

        # ── polygon-draw state ────────────────────────────────────────────────
        self._poly_points: list[tuple[float, float]] = []  # normalised pts
        self._poly_items: list[int] = []  # canvas item IDs
        self._polygons: list[PolygonAnnotation] = []  # committed polys

        # ── semantic segmentation state ───────────────────────────────────────
        # Maps class_name → hex colour string, set by the segmentation panel
        self._class_color_map: dict[str, str] = {}
        # Mask fill opacity 0.0–1.0; used for stipple selection
        self._poly_opacity: float = 0.40
        # Colour used for the in-progress polygon preview
        self._active_poly_color: str = "#00ff88"

        # ── playback state ────────────────────────────────────────────────────
        self._playing = False
        self._play_interval = 150  # ms between frames during auto-play
        self._play_job = None  # after() job id

        # frame→canvas offset (for aspect-ratio letterboxing)
        self._frame_offset_x = 0
        self._frame_offset_y = 0
        self._frame_scale = 1.0
        self._frame_w = 1
        self._frame_h = 1

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # ── mode toggle bar ───────────────────────────────────────────────────
        mode_bar = tk.Frame(self, bg=BG_PANEL, height=32)
        mode_bar.pack(fill=tk.X)
        mode_bar.pack_propagate(False)

        tk.Label(
            mode_bar, text="  MODE:", bg=BG_PANEL, fg=TEXT_LIGHT,
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(8, 4), pady=6)

        self._view_btn = tk.Button(
            mode_bar, text="👁 View", command=self.set_view_mode,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=10, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self._view_btn.pack(side=tk.LEFT, padx=2, pady=4)

        self._draw_btn = tk.Button(
            mode_bar, text="✏ Draw Box", command=self.set_draw_mode,
            bg=BG_DARK, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=10, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self._draw_btn.pack(side=tk.LEFT, padx=2, pady=4)

        self._poly_btn = tk.Button(
            mode_bar, text="⬠ Polygon", command=self.set_polygon_mode,
            bg=BG_DARK, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=10, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self._poly_btn.pack(side=tk.LEFT, padx=2, pady=4)

        self._mode_label = tk.Label(
            mode_bar, text="  VIEW MODE — navigate freely",
            bg=BG_PANEL, fg="#aaaacc", font=("Consolas", 8, "italic"),
        )
        self._mode_label.pack(side=tk.LEFT, padx=12)

        # ── canvas ────────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._on_canvas_resize())

        # mouse events (always bound — gated by mode inside handlers)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Escape>", lambda _: self._cancel_polygon())
        self.canvas.bind("<space>", lambda _: self._toggle_play())
        self.canvas.bind("<FocusIn>", lambda _: None)  # allow key events

        self._draw_empty_hint()

        # ── controls ──────────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_PANEL)
        ctrl.pack(fill=tk.X, pady=(2, 0))

        self.slider = tk.Scale(
            ctrl, from_=1, to=1, orient=tk.HORIZONTAL,
            command=self._on_slider,
            bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
            highlightthickness=0, sliderrelief=tk.FLAT,
        )
        self.slider.pack(fill=tk.X, padx=8)

        btns = tk.Frame(ctrl, bg=BG_PANEL)
        btns.pack(pady=4)

        for text, cmd in [("⏮", self._go_first), ("⏪", self._prev)]:
            tk.Button(
                btns, text=text, command=cmd,
                bg=ACCENT, fg="white", relief=tk.FLAT,
                width=4, font=("Consolas", 11), cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        # ── centred pause / play button ───────────────────────────────────────
        self._pause_btn = tk.Button(
            btns, text="▶",
            command=self._toggle_play,
            bg=ACCENT, fg="white",
            activebackground="#9d8fff",
            activeforeground="white",
            relief=tk.FLAT,
            width=5, font=("Consolas", 13, "bold"), cursor="hand2",
        )
        self._pause_btn.pack(side=tk.LEFT, padx=6)

        for text, cmd in [("⏩", self._next), ("⏭", self._go_last)]:
            tk.Button(
                btns, text=text, command=cmd,
                bg=ACCENT, fg="white", relief=tk.FLAT,
                width=4, font=("Consolas", 11), cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        # Speed selector
        speed_frame = tk.Frame(ctrl, bg=BG_PANEL)
        speed_frame.pack(pady=(0, 2))
        tk.Label(
            speed_frame, text="Speed:",
            bg=BG_PANEL, fg="#8888aa", font=("Consolas", 8),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._speed_btns: dict[int, tk.Button] = {}
        for label, ms in [("0.5×", 300), ("1×", 150), ("2×", 75), ("4×", 38)]:
            b = tk.Button(
                speed_frame, text=label,
                command=lambda m=ms: self._set_speed(m),
                bg=BG_DARK, fg="#8888aa",
                relief=tk.FLAT, padx=6, pady=1,
                font=("Consolas", 8), cursor="hand2",
            )
            b.pack(side=tk.LEFT, padx=1)
            self._speed_btns[ms] = b
        self._speed_btns[150].config(bg=ACCENT, fg="white")  # default 1×

        self.idx_label = tk.Label(
            ctrl, text="Frame —",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 9),
        )
        self.idx_label.pack(pady=(0, 4))

    # ── public API ────────────────────────────────────────────────────────────
    def load(self, loader, indices: list[int],
             frame_path_provider: Callable[[int], str] | None = None):
        self._loader = loader
        self._indices = indices
        self._pos = 0
        self._frame_path_provider = frame_path_provider
        self._clear_hint()
        if indices:
            self.slider.config(to=len(indices))
            self._show_current()

    def set_overlay_boxes(self, boxes: list[BoundingBox]):
        self._boxes = list(boxes)
        if not self._selected_is_suggestion and self._selected_idx is not None and self._selected_idx >= len(self._boxes):
            self._selected_idx = None
        self._redraw()

    def set_overlay_suggested_boxes(self, suggestions: list[BoundingBox]):
        self._suggested_boxes = list(suggestions)
        if self._selected_is_suggestion and self._selected_idx is not None and self._selected_idx >= len(self._suggested_boxes):
            self._selected_idx = None
        self._redraw()

    def set_overlay_polygons(self, polygons: list[PolygonAnnotation]):
        """Display committed polygon overlays on the canvas."""
        self._polygons = list(polygons)
        self._redraw()

    def set_overlay_suggested_polygons(self, suggestions: list[PolygonAnnotation]):
        """Display AI suggested polygon overlays on the canvas."""
        self._suggested_polygons = list(suggestions)
        self._redraw()

    def set_class_color_map(self, color_map: dict[str, str]) -> None:
        """
        Tell the canvas which hex colour to use for each class name.
        e.g. {"road": "#E74C3C", "sky": "#3498DB"}
        """
        self._class_color_map = color_map
        self._redraw()

    def set_active_poly_color(self, color: str) -> None:
        """Set the colour used for the in-progress polygon preview."""
        self._active_poly_color = color

    def set_poly_opacity(self, opacity: float) -> None:
        """Set mask fill opacity (0.0 = outline only, 1.0 = solid fill)."""
        self._poly_opacity = max(0.0, min(1.0, opacity))
        self._redraw()

    def set_selected_box(self, idx: int | None, is_suggestion: bool = False):
        """Public — used by AnnotationPanel to sync selection from list."""
        target_list = self._suggested_boxes if is_suggestion else self._boxes
        if idx is not None and (idx < 0 or idx >= len(target_list)):
            idx = None
        self._select_box(idx, is_suggestion=is_suggestion)

    @property
    def current_frame_index(self) -> int:
        if not self._indices:
            return 0
        return self._indices[self._pos]

    # ── mode switching ────────────────────────────────────────────────────────
    def set_view_mode(self):
        self._mode = self.MODE_VIEW
        self.canvas.config(cursor="arrow")
        self._view_btn.config(bg=ACCENT, fg="white")
        self._draw_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._poly_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._mode_label.config(text="  VIEW MODE — navigate freely")
        self._cancel_draw()
        self._cancel_polygon()
        if self._on_mode_change:
            self._on_mode_change("view")

    def set_draw_mode(self):
        self._mode = self.MODE_DRAW
        self.canvas.config(cursor="crosshair")
        self._draw_btn.config(bg="#e05c5c", fg="white")
        self._view_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._poly_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._mode_label.config(text="  DRAW MODE — click & drag to annotate")
        self._cancel_polygon()
        if self._on_mode_change:
            self._on_mode_change("draw")

    def set_polygon_mode(self):
        self._mode = self.MODE_POLYGON
        self.canvas.config(cursor="crosshair")
        self._poly_btn.config(bg="#2a9d5c", fg="white")
        self._view_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._draw_btn.config(bg=BG_DARK, fg=TEXT_LIGHT)
        self._mode_label.config(
            text="  POLYGON MODE — click to add pts · double-click to close"
        )
        self._cancel_draw()
        if self._on_mode_change:
            self._on_mode_change("polygon")

    def is_draw_mode(self) -> bool:
        return self._mode == self.MODE_DRAW

    def is_polygon_mode(self) -> bool:
        return self._mode == self.MODE_POLYGON

    # ── mouse handlers ────────────────────────────────────────────────────────
    def _on_mouse_press(self, event):
        if self._mode == self.MODE_DRAW:
            self._draw_start = (event.x, event.y)
            if self._draw_rect:
                self.canvas.delete(self._draw_rect)
            self._draw_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#ff4444", width=2, dash=(4, 4),
            )
            return

        if self._mode == self.MODE_POLYGON:
            # Double-click closes the polygon
            if event.num == 1 and getattr(event, "type", None) is not None:
                pass  # handled via ButtonPress; double fires both
            norm = self._canvas_pt_to_norm(event.x, event.y)
            if norm:
                self._poly_points.append(norm)
                self._draw_poly_preview()
            return

        # View mode → box editing
        # If a box is already selected and we pressed near a handle, start
        # resizing. Otherwise, see if the press hit any box → select it.
        if self._selected_idx is not None:
            handle = self._handle_at(event.x, event.y, self._selected_idx, is_suggestion=self._selected_is_suggestion)
            if handle is not None:
                self._begin_edit(handle, event.x, event.y)
                return

        hit = self._box_at(event.x, event.y)
        if hit is not None:
            hit_idx, is_sugg = hit
            self._select_box(hit_idx, is_suggestion=is_sugg)
            self._begin_edit("move", event.x, event.y)
        else:
            # Click on empty area → deselect
            self._select_box(None, is_suggestion=False)

    def _on_mouse_drag(self, event):
        if self._mode == self.MODE_DRAW and self._draw_start is not None:
            x0, y0 = self._draw_start
            self.canvas.coords(self._draw_rect, x0, y0, event.x, event.y)
            return

        # Edit drag
        if self._edit_handle is not None and self._selected_idx is not None:
            self._apply_edit_drag(event.x, event.y, commit=False)

    def _on_double_click(self, event):
        if self._mode == self.MODE_POLYGON:
            self._close_polygon()

    def _on_mouse_release(self, event):
        if self._mode == self.MODE_DRAW and self._draw_start is not None:
            x0, y0 = self._draw_start
            x1, y1 = event.x, event.y
            if self._draw_rect:
                self.canvas.delete(self._draw_rect)
                self._draw_rect = None
            self._draw_start = None
            if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
                return
            norm = self._canvas_to_norm(x0, y0, x1, y1)
            if norm and self._on_box_drawn:
                self._on_box_drawn(*norm)
            return

        # Edit release → commit final coords to caller
        if self._edit_handle is not None and self._selected_idx is not None:
            self._apply_edit_drag(event.x, event.y, commit=True)
            self._edit_handle = None
            self._edit_drag_start = None

    def _cancel_draw(self):
        if self._draw_rect:
            self.canvas.delete(self._draw_rect)
            self._draw_rect = None
        self._draw_start = None

    def _cancel_polygon(self):
        for item in self._poly_items:
            self.canvas.delete(item)
        self._poly_items.clear()
        self._poly_points.clear()

    def _draw_poly_preview(self):
        for item in self._poly_items:
            self.canvas.delete(item)
        self._poly_items.clear()
        pts = self._poly_points
        if not pts:
            return
        # Convert normalised pts → canvas coords
        cv = [self._norm_to_canvas(x, y) for x, y in pts]
        col = self._active_poly_color
        for cx, cy in cv:
            dot = self.canvas.create_oval(
                cx - 4, cy - 4, cx + 4, cy + 4,
                fill=col, outline="white", width=1,
            )
            self._poly_items.append(dot)
        if len(cv) >= 2:
            flat = [c for pt in cv for c in pt]
            line = self.canvas.create_line(
                *flat, fill=col, width=2, tags=("overlay",)
            )
            self._poly_items.append(line)
        # Close-line hint
        if len(cv) >= 3:
            close = self.canvas.create_line(
                cv[-1][0], cv[-1][1], cv[0][0], cv[0][1],
                fill=col, width=1, dash=(4, 4), tags=("overlay",)
            )
            self._poly_items.append(close)

    def _close_polygon(self):
        """Finalise the in-progress polygon and fire the callback."""
        if len(self._poly_points) < 3:
            return
        pts = list(self._poly_points)
        self._cancel_polygon()
        if self._on_polygon_drawn:
            self._on_polygon_drawn(pts)
        self._redraw()

    def _canvas_pt_to_norm(self, cx: float, cy: float):
        """Convert a single canvas point to normalised (x, y) in [0,1]."""
        if self._frame_scale == 0:
            return None
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc = self._frame_scale
        x = max(0.0, min((cx - ox) / sc / self._frame_w, 1.0))
        y = max(0.0, min((cy - oy) / sc / self._frame_h, 1.0))
        return x, y

    def _norm_to_canvas(self, nx: float, ny: float) -> tuple[float, float]:
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc = self._frame_scale
        return ox + nx * self._frame_w * sc, oy + ny * self._frame_h * sc

    # ── box editing helpers ───────────────────────────────────────────────────
    HANDLE_SIZE = 6  # pixels — square half-side for resize hit-test

    def _box_pixel_rect(self, box):
        """Return (x1, y1, x2, y2) in canvas (not frame) coordinates."""
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc = self._frame_scale
        cx = box.x_center * self._frame_w
        cy = box.y_center * self._frame_h
        w = box.width * self._frame_w
        h = box.height * self._frame_h
        x1 = ox + (cx - w / 2) * sc
        y1 = oy + (cy - h / 2) * sc
        x2 = ox + (cx + w / 2) * sc
        y2 = oy + (cy + h / 2) * sc
        return x1, y1, x2, y2

    def _box_at(self, cx, cy) -> tuple[int, bool] | None:
        """Return (index, is_suggestion) of topmost box containing canvas point (cx,cy), or None."""
        candidates = []
        for i, b in enumerate(self._boxes):
            x1, y1, x2, y2 = self._box_pixel_rect(b)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                candidates.append((i, False, (x2 - x1) * (y2 - y1)))
        for i, b in enumerate(self._suggested_boxes):
            x1, y1, x2, y2 = self._box_pixel_rect(b)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                candidates.append((i, True, (x2 - x1) * (y2 - y1)))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[2])
        return candidates[0][0], candidates[0][1]

    def _handle_at(self, cx, cy, box_idx, is_suggestion: bool = False) -> str | None:
        target_list = self._suggested_boxes if is_suggestion else self._boxes
        if box_idx is None or box_idx >= len(target_list):
            return None
        x1, y1, x2, y2 = self._box_pixel_rect(target_list[box_idx])
        s = self.HANDLE_SIZE
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        handles = {
            "nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
            "w": (x1, my), "e": (x2, my),
            "sw": (x1, y2), "s": (mx, y2), "se": (x2, y2),
        }
        for name, (hx, hy) in handles.items():
            if abs(cx - hx) <= s and abs(cy - hy) <= s:
                return name
        return None

    def _begin_edit(self, handle, mx, my):
        if self._selected_idx is None:
            return
        target_list = self._suggested_boxes if self._selected_is_suggestion else self._boxes
        if self._selected_idx >= len(target_list):
            return
        box = target_list[self._selected_idx]
        self._edit_handle = handle
        self._edit_drag_start = (mx, my, self._box_pixel_rect(box))

    def _apply_edit_drag(self, mx, my, commit: bool):
        if self._edit_drag_start is None or self._selected_idx is None:
            return
        sx, sy, (x1, y1, x2, y2) = self._edit_drag_start
        dx, dy = mx - sx, my - sy
        h = self._edit_handle

        if h == "move":
            x1 += dx
            y1 += dy
            x2 += dx
            y2 += dy
        else:
            if "n" in h:
                y1 += dy
            if "s" in h:
                y2 += dy
            if "w" in h:
                x1 += dx
            if "e" in h:
                x2 += dx

        # Clamp to frame area
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc = self._frame_scale
        fx2 = ox + self._frame_w * sc
        fy2 = oy + self._frame_h * sc
        x1, x2 = sorted((max(ox, min(x1, fx2)), max(ox, min(x2, fx2))))
        y1, y2 = sorted((max(oy, min(y1, fy2)), max(oy, min(y2, fy2))))

        norm = self._canvas_to_norm(x1, y1, x2, y2)
        if norm is None:
            return
        nx1, ny1, nx2, ny2 = norm
        if nx2 - nx1 < 0.005 or ny2 - ny1 < 0.005:
            return  # ignore degenerate

        # Live-preview by updating local copy + redraw
        target_list = self._suggested_boxes if self._selected_is_suggestion else self._boxes
        if self._selected_idx >= len(target_list):
            return
        box = target_list[self._selected_idx]
        box.x_center = (nx1 + nx2) / 2
        box.y_center = (ny1 + ny2) / 2
        box.width = nx2 - nx1
        box.height = ny2 - ny1
        self._redraw()

        if commit and self._on_box_edited:
            try:
                self._on_box_edited(self._selected_idx, nx1, ny1, nx2, ny2, self._selected_is_suggestion)
            except TypeError:
                self._on_box_edited(self._selected_idx, nx1, ny1, nx2, ny2)

    def _select_box(self, idx: int | None, is_suggestion: bool = False):
        if idx == self._selected_idx and is_suggestion == self._selected_is_suggestion:
            return
        self._selected_idx = idx
        self._selected_is_suggestion = is_suggestion
        self._redraw()
        if self._on_box_selected:
            try:
                self._on_box_selected(idx, is_suggestion)
            except TypeError:
                self._on_box_selected(idx)

    # ── coord conversion ──────────────────────────────────────────────────────
    def _canvas_to_norm(self, cx0, cy0, cx1, cy1):
        """
        Convert canvas pixel rect → (x1_n, y1_n, x2_n, y2_n) normalised to
        the *original frame* dimensions. Returns None if frame not loaded.
        """
        if self._frame_scale == 0:
            return None

        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc = self._frame_scale

        # Clamp to frame area
        ix0 = max(0.0, min((cx0 - ox) / sc, self._frame_w))
        iy0 = max(0.0, min((cy0 - oy) / sc, self._frame_h))
        ix1 = max(0.0, min((cx1 - ox) / sc, self._frame_w))
        iy1 = max(0.0, min((cy1 - oy) / sc, self._frame_h))

        # Normalise to [0, 1]
        nx0 = ix0 / self._frame_w
        ny0 = iy0 / self._frame_h
        nx1 = ix1 / self._frame_w
        ny1 = iy1 / self._frame_h

        # Ensure top-left < bottom-right
        return (min(nx0, nx1), min(ny0, ny1),
                max(nx0, nx1), max(ny0, ny1))

    # ── navigation ────────────────────────────────────────────────────────────
    def _go_first(self):
        self._stop_play()
        self._goto(0)

    def _go_last(self):
        self._stop_play()
        self._goto(len(self._indices) - 1)

    def _prev(self):
        self._stop_play()
        self._goto(self._pos - 1)

    def _next(self):
        self._goto(self._pos + 1)

    # ── playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        if not self._indices:
            return
        self._playing = True
        self._pause_btn.config(text="⏸", bg="#ff6584", fg="white")
        self._tick()

    def _stop_play(self) -> None:
        self._playing = False
        self._pause_btn.config(text="▶", bg=ACCENT, fg="white")
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None

    def _tick(self) -> None:
        if not self._playing:
            return
        # Stop at last frame instead of wrapping
        if self._pos >= len(self._indices) - 1:
            self._stop_play()
            return
        self._goto(self._pos + 1)
        self._play_job = self.after(self._play_interval, self._tick)

    def _set_speed(self, interval_ms: int) -> None:
        self._play_interval = interval_ms
        for ms, btn in self._speed_btns.items():
            if ms == interval_ms:
                btn.config(bg=ACCENT, fg="white")
            else:
                btn.config(bg=BG_DARK, fg="#8888aa")
        if self._playing:
            if self._play_job is not None:
                self.after_cancel(self._play_job)
            self._play_job = self.after(self._play_interval, self._tick)

    def _on_slider(self, val):
        self._goto(int(val) - 1, update_slider=False)

    def _goto(self, pos: int, update_slider: bool = True):
        if not self._indices:
            return
        self._pos = max(0, min(pos, len(self._indices) - 1))
        if update_slider:
            self.slider.set(self._pos + 1)
        self._show_current()

    def _show_current(self):
        if not self._loader or not self._indices:
            return
        idx = self._indices[self._pos]
        frame = self._read_frame_reliable(idx)
        if frame is None:
            return
        self._current_frame = frame
        self._frame_h, self._frame_w = frame.shape[:2]
        self.idx_label.config(
            text=f"Frame {idx + 1}  ({self._pos + 1} / {len(self._indices)})"
        )
        if self._on_change:
            self._on_change(idx, frame)
        self._redraw()

    def _on_canvas_resize(self):
        if self._current_frame is None:
            self._draw_empty_hint()
        else:
            self._redraw()

    def _read_frame_reliable(self, idx: int):
        """
        Prefer the saved PNG over re-seeking the video — cv2 seek is
        unreliable on H.264/variable-rate videos and can return wrong
        frames or None mid-navigation.
        """
        if self._frame_path_provider:
            try:
                path = self._frame_path_provider(idx)
            except Exception:
                path = None
            if path and os.path.exists(path):
                img = safe_imread(path)
                if img is not None:
                    return img
        return self._loader.read_frame(idx)

    # ── empty-state hint ──────────────────────────────────────────────────────
    def _draw_empty_hint(self):
        self.canvas.delete("hint")
        cw = self.canvas.winfo_width() or 640
        ch = self.canvas.winfo_height() or 480
        cx, cy = cw // 2, ch // 2
        self.canvas.create_text(
            cx, cy - 28, text="📂",
            font=("Consolas", 36), fill="#3a3a5e", tags="hint",
        )
        self.canvas.create_text(
            cx, cy + 20,
            text="Click to open a video or image",
            font=("Consolas", 13), fill="#4a4a6e", tags="hint",
        )
        self.canvas.create_text(
            cx, cy + 44,
            text="— or use the toolbar buttons above —",
            font=("Consolas", 9), fill="#333355", tags="hint",
        )
        self.canvas.tag_bind("hint", "<Button-1>", self._on_hint_click)
        self.canvas.config(cursor="hand2")

    def _clear_hint(self):
        self.canvas.delete("hint")
        self.canvas.delete("image")
        self.canvas.delete("overlay")
        self._image_id = None
        self._photo = None
        self._photo_size = (0, 0)
        self.canvas.config(cursor="arrow")

    def _on_hint_click(self, _event):
        if self._on_open_request:
            self._on_open_request()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _redraw(self):
        if self._current_frame is None:
            return

        cw = self.canvas.winfo_width() or 640
        ch = self.canvas.winfo_height() or 480
        fh, fw = self._current_frame.shape[:2]

        # Compute letterbox scale + offsets
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2

        self._frame_scale = scale
        self._frame_offset_x = ox
        self._frame_offset_y = oy
        self._frame_w = fw
        self._frame_h = fh

        frame = self._current_frame
        if self._boxes:
            from utils.image_utils import draw_boxes
            frame = draw_boxes(frame, self._boxes)

        current_frame_id = id(frame)
        if (self._photo is None
                or self._photo_size != (cw, ch)
                or self._last_frame_id != current_frame_id):
            self._photo = bgr_to_photoimage(frame, cw, ch)
            self._photo_size = (cw, ch)
            self._last_frame_id = current_frame_id
            if self._image_id is None:
                self._image_id = self.canvas.create_image(
                    cw // 2, ch // 2,
                    image=self._photo, anchor=tk.CENTER,
                    tags=("image",),
                )
            else:
                self.canvas.itemconfig(self._image_id, image=self._photo)
        else:
            if self._image_id is None:
                self._image_id = self.canvas.create_image(
                    cw // 2, ch // 2,
                    image=self._photo, anchor=tk.CENTER,
                    tags=("image",),
                )
            else:
                self.canvas.coords(self._image_id, cw // 2, ch // 2)

        self.canvas.delete("overlay")

        # Draw AI Suggested Boxes
        for si, box in enumerate(self._suggested_boxes):
            x1, y1, x2, y2 = self._box_pixel_rect(box)
            is_sel = (self._selected_is_suggestion and self._selected_idx == si)
            outline_col = "#ffaa00" if is_sel else "#aa66ff"
            dash_pat = (2, 2) if is_sel else (4, 4)
            w_size = 3 if is_sel else 2

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=outline_col, width=w_size, dash=dash_pat,
                tags=("overlay", "suggested_box"),
            )

            badge_text = f"🤖 {box.class_name} {box.confidence:.2f}"
            bw = len(badge_text) * 7 + 8
            self.canvas.create_rectangle(
                x1, max(0, y1 - 18), x1 + bw, y1,
                fill="#5a3d99" if not is_sel else "#aa6600", outline="",
                tags=("overlay", "suggested_box"),
            )
            self.canvas.create_text(
                x1 + 4, max(9, y1 - 9), text=badge_text, anchor=tk.W,
                fill="white", font=("Consolas", 8, "bold"),
                tags=("overlay", "suggested_box"),
            )

        # Highlight selected box (confirmed or suggestion) + draw resize handles on top
        target_list = self._suggested_boxes if self._selected_is_suggestion else self._boxes
        if (self._selected_idx is not None and self._selected_idx < len(target_list)):
            x1, y1, x2, y2 = self._box_pixel_rect(target_list[self._selected_idx])
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="#ffaa00", width=2, dash=(2, 2), tags=("overlay", "selection"),
            )
            s = self.HANDLE_SIZE
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            for hx, hy in [
                (x1, y1), (mx, y1), (x2, y1),
                (x1, my), (x2, my),
                (x1, y2), (mx, y2), (x2, y2),
            ]:
                self.canvas.create_rectangle(
                    hx - s, hy - s, hx + s, hy + s,
                    fill="#ffaa00", outline="white", width=1,
                    tags=("overlay", "handle"),
                )

        # Draw AI Suggested Polygons
        _FALLBACK_COLORS = [
            "#00ff88", "#ff6644", "#44aaff", "#ffcc00",
            "#cc44ff", "#ff44aa", "#44ffcc",
        ]
        for spi, poly in enumerate(self._suggested_polygons):
            color = self._class_color_map.get(
                poly.class_name,
                _FALLBACK_COLORS[spi % len(_FALLBACK_COLORS)],
            )
            cv_pts = [self._norm_to_canvas(x, y) for x, y in poly.points]
            if len(cv_pts) >= 3:
                flat = [c for pt in cv_pts for c in pt]
                self.canvas.create_polygon(
                    *flat,
                    outline=color, fill=color, stipple="gray25",
                    width=2, dash=(4, 4), tags=("overlay", "suggested_polygon"),
                )
                self.canvas.create_polygon(
                    *flat,
                    outline=color, fill="", width=2, dash=(4, 4),
                    tags=("overlay", "suggested_polygon"),
                )
            if cv_pts:
                cx_c = sum(p[0] for p in cv_pts) / len(cv_pts)
                cy_c = sum(p[1] for p in cv_pts) / len(cv_pts)
                badge_text = f"🤖 {poly.class_name} {poly.confidence:.2f}"
                self.canvas.create_text(
                    cx_c, cy_c, text=badge_text,
                    fill=color, font=("Helvetica", 8, "bold"),
                    tags=("overlay", "suggested_polygon"),
                )

        # Draw committed polygon overlays (semantic colours + opacity)
        # Choose stipple pattern based on opacity setting
        opacity = self._poly_opacity
        if opacity >= 0.75:
            stipple = "gray75"
        elif opacity >= 0.50:
            stipple = "gray50"
        elif opacity >= 0.25:
            stipple = "gray25"
        else:
            stipple = "gray12"

        for pi, poly in enumerate(self._polygons):
            # Use the class colour from the panel; fall back to palette
            color = self._class_color_map.get(
                poly.class_name,
                _FALLBACK_COLORS[pi % len(_FALLBACK_COLORS)],
            )
            cv_pts = [self._norm_to_canvas(x, y) for x, y in poly.points]
            if len(cv_pts) >= 3:
                flat = [c for pt in cv_pts for c in pt]
                # Filled mask with selected opacity
                self.canvas.create_polygon(
                    *flat,
                    outline=color, fill=color, stipple=stipple,
                    width=2, tags=("overlay", "polygon"),
                )
                # Solid outline on top so the border is always crisp
                self.canvas.create_polygon(
                    *flat,
                    outline=color, fill="", width=2,
                    tags=("overlay", "polygon"),
                )
            # Class label at centroid
            if cv_pts:
                cx_c = sum(p[0] for p in cv_pts) / len(cv_pts)
                cy_c = sum(p[1] for p in cv_pts) / len(cv_pts)
                # Dark shadow for readability
                self.canvas.create_text(
                    cx_c + 1, cy_c + 1, text=poly.class_name,
                    fill="#000000", font=("Helvetica", 8, "bold"),
                    tags=("overlay", "polygon"),
                )
                self.canvas.create_text(
                    cx_c, cy_c, text=poly.class_name,
                    fill="white", font=("Helvetica", 8, "bold"),
                    tags=("overlay", "polygon"),
                )

        # Polygon preview (in-progress)
        self._draw_poly_preview()
