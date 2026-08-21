"""
ui/segmentation_panel.py
──────────────────────────────────────────────────────────────────────────────
Semantic Segmentation panel — right-hand panel shown when annotation type
is 'Segmentation'.

Features added (on top of the original polygon list):
  • Colour-coded class list with swatches
  • Add / rename / delete semantic class
  • Opacity slider for filled mask overlays
  • Per-frame polygon list (class name + vertex count)
  • Delete selected polygon
  • Save / Clear frame actions
"""

import tkinter as tk
from collections.abc import Callable
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

from models.annotation_model import PolygonAnnotation
from utils.config import (
    ACCENT,
    BG_DARK,
    BG_PANEL,
    TEXT_LIGHT,
    YOLO_DEFAULT_SEG_MODEL,
    YOLO_SEG_MODELS,
)


def _hover_btn(btn, normal, hover):
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda _e: btn.config(bg=normal))

# ── default colour palette ─────────────────────────────────────────────────
_PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#E91E63", "#F1C40F", "#00BCD4",
]

ROW_H = 18   # approximate Listbox row height in pixels


class SegmentationPanel(tk.Frame):
    """
    Semantic segmentation right-panel.

    Callbacks (all optional):
        on_save_click()
        on_clear_click()
        on_delete_poly(index: int)
        on_poly_select(index: int | None)
        on_class_changed(class_name: str, color: str)
        on_opacity_change(value: float)   # 0.0 – 1.0
        on_model_change(model_name: str)
    """

    def __init__(
        self,
        master,
        on_save_click:         Callable = None,
        on_clear_click:        Callable = None,
        on_delete_poly:        Callable = None,
        on_poly_select:        Callable = None,
        on_class_changed:      Callable = None,
        on_opacity_change:     Callable = None,
        on_model_change:       Callable = None,
        on_auto_seg_click:     Callable = None,
        on_auto_seg_all_click: Callable = None,
    ) -> None:
        super().__init__(master, bg=BG_PANEL, width=280)
        self.pack_propagate(False)

        self._on_save          = on_save_click
        self._on_clear         = on_clear_click
        self._on_delete_poly   = on_delete_poly
        self._on_poly_select   = on_poly_select
        self._on_class_changed = on_class_changed
        self._on_opacity       = on_opacity_change
        self._on_model_change  = on_model_change
        self._on_auto_seg      = on_auto_seg_click
        self._on_auto_seg_all  = on_auto_seg_all_click

        # semantic class list: [{name, color}, …]
        self._classes: list[dict] = []
        self._selected_class_idx: int | None = None

        self._build()
        # seed one default class
        self._add_class(name="object", color=_PALETTE[0], notify=False)

    # ── build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 3}

        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            hdr, text="SEGMENTATION",
            bg=BG_PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"),
        ).pack(side=tk.LEFT, padx=10)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(4, 0))

        self._tips_open = False
        tips_row = tk.Frame(self, bg=BG_PANEL)
        tips_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._tips_btn = tk.Button(
            tips_row, text="ℹ  How to annotate  ▸",
            bg=BG_DARK, fg="#8888aa", relief=tk.FLAT,
            font=("Consolas", 8), cursor="hand2",
            activebackground=BG_PANEL, activeforeground=TEXT_LIGHT, bd=0,
            anchor=tk.W, command=self._toggle_tips,
        )
        self._tips_btn.pack(fill=tk.X, ipady=2)
        self._tips_label = tk.Label(
            self,
            text=(
                "  1. Pick a semantic class below\n"
                "  2. Click '⬠ Polygon' mode button\n"
                "  3. Click canvas to place vertices\n"
                "  4. Double-click to close polygon\n"
                "  5. Press Esc to cancel in-progress"
            ),
            bg=BG_PANEL, fg="#8888aa",
            font=("Consolas", 8), justify=tk.LEFT,
        )

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # ── Model Selection for Auto Polygon ──────────────────────────────
        tk.Label(
            self, text="MODEL",
            bg=BG_PANEL, fg="#888899", font=("Consolas", 7, "bold"),
        ).pack(pady=(2, 2))

        model_row = tk.Frame(self, bg=BG_PANEL)
        model_row.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.model_var = tk.StringVar(value=YOLO_DEFAULT_SEG_MODEL)
        self._model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var,
            values=YOLO_SEG_MODELS, font=("Consolas", 8), state="readonly", width=14,
        )
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        self._model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_selected())

        browse_btn = tk.Button(
            model_row, text="📂",
            command=self._browse_model,
            bg=BG_DARK, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=5, font=("Consolas", 9), cursor="hand2",
            activebackground=ACCENT, activeforeground="white", bd=0,
        )
        browse_btn.pack(side=tk.LEFT, padx=(4, 0))
        _hover_btn(browse_btn, BG_DARK, ACCENT)

        op_row = tk.Frame(self, bg=BG_PANEL)
        op_row.pack(fill=tk.X, **pad)
        tk.Label(op_row, text="Mask opacity:", bg=BG_PANEL, fg=TEXT_LIGHT,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._opacity_var = tk.DoubleVar(value=0.40)
        tk.Scale(
            op_row,
            variable=self._opacity_var,
            from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=110,
            bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
            highlightthickness=0, bd=0,
            command=lambda v: self._on_opacity and self._on_opacity(float(v)),
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)

        tk.Label(
            self, text="SEMANTIC CLASSES",
            bg=BG_PANEL, fg="#888899", font=("Consolas", 7, "bold"),
        ).pack(pady=(4, 2))

        cls_outer = tk.Frame(self, bg=BG_PANEL)
        cls_outer.pack(fill=tk.X, padx=8)

        self._swatch_canvas = tk.Canvas(
            cls_outer, width=16, bg=BG_DARK,
            bd=0, highlightthickness=0, height=90,
        )
        self._swatch_canvas.pack(side=tk.LEFT, fill=tk.Y)

        cls_sb = tk.Scrollbar(cls_outer, orient=tk.VERTICAL, bg=BG_DARK)
        cls_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._cls_listbox = tk.Listbox(
            cls_outer,
            yscrollcommand=cls_sb.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 9), relief=tk.FLAT, bd=0, height=5,
            activestyle="none",
        )
        self._cls_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_sb.config(command=self._cls_listbox.yview)
        self._cls_listbox.bind("<<ListboxSelect>>", self._on_cls_listbox_select)

        cls_btn_row = tk.Frame(self, bg=BG_PANEL)
        cls_btn_row.pack(fill=tk.X, padx=8, pady=(4, 2))

        for text, cmd, bg, hover in [
            ("+ Add",    self._prompt_add_class, ACCENT,    "#9d8fff"),
            ("✎ Rename", self._rename_class,     ACCENT,    "#9d8fff"),
            ("🎨 Color", self._pick_color,        ACCENT,    "#9d8fff"),
            ("✕ Del",    self._delete_class,      "#7a3333", "#a04040"),
        ]:
            b = tk.Button(
                cls_btn_row, text=text, command=cmd,
                bg=bg, fg="white", relief=tk.FLAT,
                padx=5, pady=2, font=("Consolas", 8), cursor="hand2",
                activebackground=hover, activeforeground="white", bd=0,
            )
            b.pack(side=tk.LEFT, padx=2)
            _hover_btn(b, bg, hover)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        poly_hdr = tk.Frame(self, bg=BG_PANEL)
        poly_hdr.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            poly_hdr, text="POLYGONS ON THIS FRAME",
            bg=BG_PANEL, fg="#888899", font=("Consolas", 7, "bold"),
        ).pack(side=tk.LEFT)
        self._stats_var = tk.StringVar(value="0 polygons")
        tk.Label(
            poly_hdr, textvariable=self._stats_var,
            bg=BG_PANEL, fg=ACCENT, font=("Consolas", 7, "bold"),
        ).pack(side=tk.RIGHT)

        poly_frame = tk.Frame(self, bg=BG_PANEL)
        poly_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        poly_sb = tk.Scrollbar(poly_frame)
        poly_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._poly_listbox = tk.Listbox(
            poly_frame,
            yscrollcommand=poly_sb.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 8), relief=tk.FLAT, bd=0, height=7,
        )
        self._poly_listbox.pack(fill=tk.BOTH, expand=True)
        self._poly_listbox.bind("<<ListboxSelect>>", self._on_poly_listbox_select)
        poly_sb.config(command=self._poly_listbox.yview)

        del_row = tk.Frame(self, bg=BG_PANEL)
        del_row.pack(fill=tk.X, padx=8, pady=(3, 0))
        del_poly_btn = tk.Button(
            del_row, text="🗑  Delete Selected",
            command=self._delete_selected_poly,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=6, pady=3, font=("Consolas", 8), cursor="hand2",
            activebackground="#a04040", activeforeground="white", bd=0,
        )
        del_poly_btn.pack(side=tk.RIGHT)
        _hover_btn(del_poly_btn, "#7a3333", "#a04040")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(6, 4))

        auto_row = tk.Frame(self, bg=BG_PANEL)
        auto_row.pack(fill=tk.X, padx=8, pady=2)

        auto_btn = tk.Button(
            auto_row, text="⚡ Auto Polygon",
            command=lambda: self._on_auto_seg and self._on_auto_seg(),
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=6, pady=5, font=("Consolas", 8, "bold"), cursor="hand2",
            activebackground="#9d8fff", activeforeground="white", bd=0,
        )
        auto_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        _hover_btn(auto_btn, ACCENT, "#9d8fff")

        auto_all_btn = tk.Button(
            auto_row, text="⚡ Auto All",
            command=lambda: self._on_auto_seg_all and self._on_auto_seg_all(),
            bg="#4a3a8a", fg="white", relief=tk.FLAT,
            padx=6, pady=5, font=("Consolas", 8, "bold"), cursor="hand2",
            activebackground="#6a5aaf", activeforeground="white", bd=0,
        )
        auto_all_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        _hover_btn(auto_all_btn, "#4a3a8a", "#6a5aaf")

        save_btn = tk.Button(
            self, text="💾  Save Annotations",
            command=lambda: self._on_save and self._on_save(),
            bg="#2d8a4e", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#3da060", activeforeground="white", bd=0,
        )
        save_btn.pack(fill=tk.X, padx=8, pady=2)
        _hover_btn(save_btn, "#2d8a4e", "#3da060")

        clear_btn = tk.Button(
            self, text="🗑  Clear Frame Polygons",
            command=lambda: self._on_clear and self._on_clear(),
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#a04040", activeforeground="white", bd=0,
        )
        clear_btn.pack(fill=tk.X, padx=8, pady=(2, 8))
        _hover_btn(clear_btn, "#7a3333", "#a04040")

    def _toggle_tips(self):
        self._tips_open = not self._tips_open
        if self._tips_open:
            self._tips_label.pack(anchor=tk.W, padx=10, pady=(0, 2))
            self._tips_btn.config(text="ℹ  How to annotate  ▾")
        else:
            self._tips_label.pack_forget()
            self._tips_btn.config(text="ℹ  How to annotate  ▸")

    # ── public API ─────────────────────────────────────────────────────────

    def update_polygons(
        self,
        polygons: list[PolygonAnnotation],
        class_names: list[str],
        *args,
    ) -> None:
        """Refresh polygon list. Also syncs class combo values."""
        # merge any unseen class names into our class list
        existing = {c["name"] for c in self._classes}
        for name in class_names:
            if name not in existing:
                color = _PALETTE[len(self._classes) % len(_PALETTE)]
                self._add_class(name=name, color=color, notify=False)
                existing.add(name)

        self._poly_listbox.delete(0, tk.END)
        for i, poly in enumerate(polygons):
            color = self._color_for(poly.class_name)
            conf  = f"{poly.confidence:.2f}" if poly.confidence < 1.0 else "manual"
            self._poly_listbox.insert(
                tk.END,
                f"  [{i:02d}] {poly.class_name:<14} {len(poly.points):2d}pts  {conf}",
            )
            # tint the row with the class colour
            self._poly_listbox.itemconfig(
                i, fg=color, selectforeground="white",
            )

        n = len(polygons)
        self._stats_var.set(f"{n} polygon{'s' if n != 1 else ''}")

    def get_selected_class(self) -> str:
        """Return the currently highlighted class name."""
        if self._selected_class_idx is not None and self._classes:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                return self._classes[idx]["name"]
        return self._classes[0]["name"] if self._classes else "object"

    def get_selected_color(self) -> str:
        """Return the hex colour for the active class."""
        if self._selected_class_idx is not None and self._classes:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                return self._classes[idx]["color"]
        return _PALETTE[0]

    def get_opacity(self) -> float:
        return self._opacity_var.get()

    def get_model_name(self) -> str:
        return self.model_var.get()

    def set_model_name(self, name: str) -> None:
        self.model_var.set(name)

    def set_class_names(self, names: list[str]) -> None:
        """Bulk-load class names (e.g. from YOLO model)."""
        existing = {c["name"] for c in self._classes}
        for name in names:
            if name not in existing:
                color = _PALETTE[len(self._classes) % len(_PALETTE)]
                self._add_class(name=name, color=color, notify=False)
                existing.add(name)

    # ── model callbacks ────────────────────────────────────────────────────

    def _on_model_selected(self) -> None:
        if self._on_model_change:
            self._on_model_change(self.model_var.get())

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model weights",
            filetypes=[
                ("Model weights", "*.pt *.onnx"),
                ("PyTorch (Ultralytics/AGPL)", "*.pt"),
                ("ONNX (AGPL-free)", "*.onnx"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if path:
            self.model_var.set(path)
            current = list(self._model_combo["values"])
            if path not in current:
                self._model_combo["values"] = [path] + current
            if self._on_model_change:
                self._on_model_change(path)

    # ── class management ───────────────────────────────────────────────────

    def _prompt_add_class(self) -> None:
        name = simpledialog.askstring("New class", "Class name:", parent=self)
        if name and name.strip():
            self._add_class(name=name.strip())

    def _add_class(
        self,
        name: str = "object",
        color: str | None = None,
        notify: bool = True,
    ) -> None:
        if color is None:
            color = _PALETTE[len(self._classes) % len(_PALETTE)]
        self._classes.append({"name": name, "color": color})
        self._cls_listbox.insert(tk.END, f"  {name}")
        self._refresh_swatches()
        # auto-select if first
        if len(self._classes) == 1:
            self._cls_listbox.selection_set(0)
            self._selected_class_idx = 0
        if notify:
            self._fire_class_changed()

    def _rename_class(self) -> None:
        idx = self._selected_class_idx
        if idx is None:
            return
        new = simpledialog.askstring(
            "Rename class", "New name:",
            initialvalue=self._classes[idx]["name"],
            parent=self,
        )
        if new and new.strip():
            self._classes[idx]["name"] = new.strip()
            self._cls_listbox.delete(idx)
            self._cls_listbox.insert(idx, f"  {new.strip()}")
            self._cls_listbox.selection_set(idx)
            self._fire_class_changed()

    def _pick_color(self) -> None:
        idx = self._selected_class_idx
        if idx is None:
            return
        result = colorchooser.askcolor(
            color=self._classes[idx]["color"],
            title="Pick class colour",
            parent=self,
        )
        if result and result[1]:
            self._classes[idx]["color"] = result[1]
            self._refresh_swatches()
            self._fire_class_changed()

    def _delete_class(self) -> None:
        idx = self._selected_class_idx
        if idx is None or not self._classes:
            return
        if len(self._classes) == 1:
            messagebox.showinfo("Cannot delete", "At least one class is required.")
            return
        name = self._classes[idx]["name"]
        if not messagebox.askyesno(
            "Delete class",
            f"Delete class '{name}'?\n"
            "Existing polygons tagged with this class keep their label.",
            parent=self,
        ):
            return
        self._classes.pop(idx)
        self._cls_listbox.delete(idx)
        self._refresh_swatches()
        new_idx = min(idx, len(self._classes) - 1)
        self._cls_listbox.selection_set(new_idx)
        self._selected_class_idx = new_idx
        self._fire_class_changed()

    def _on_cls_listbox_select(self, _event=None) -> None:
        sel = self._cls_listbox.curselection()
        if sel:
            self._selected_class_idx = sel[0]
            self._fire_class_changed()

    def _fire_class_changed(self) -> None:
        if self._on_class_changed and self._selected_class_idx is not None:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                c = self._classes[idx]
                self._on_class_changed(c["name"], c["color"])

    def _refresh_swatches(self) -> None:
        self._swatch_canvas.delete("all")
        for i, cls in enumerate(self._classes):
            y = i * ROW_H + 4
            self._swatch_canvas.create_rectangle(
                2, y, 13, y + 11,
                fill=cls["color"], outline="",
            )

    def _color_for(self, class_name: str) -> str:
        for c in self._classes:
            if c["name"] == class_name:
                return c["color"]
        return TEXT_LIGHT

    # ── polygon list events ────────────────────────────────────────────────

    def _delete_selected_poly(self) -> None:
        sel = self._poly_listbox.curselection()
        if sel and self._on_delete_poly:
            self._on_delete_poly(sel[0])

    def _on_poly_listbox_select(self, _event) -> None:
        sel  = self._poly_listbox.curselection()
        idx  = sel[0] if sel else None
        if self._on_poly_select:
            self._on_poly_select(idx)
