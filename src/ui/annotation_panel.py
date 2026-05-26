"""
Right-hand panel with two tabs:
  AUTO  — YOLO confidence slider, run buttons
  MANUAL— class selector, box list, delete button
"""
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk

from models.annotation_model import BoundingBox
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT, YOLO_DEFAULT_MODEL, YOLO_MODELS


def _hover_btn(btn, normal, hover):
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda _e: btn.config(bg=normal))


class AnnotationPanel(tk.Frame):
    def __init__(
        self,
        master,
        on_yolo_click:       Callable,
        on_yolo_all_click:   Callable,
        on_save_click:       Callable,
        on_clear_click:      Callable,
        on_delete_box:       Callable = None,   # callable(box_index)
        on_conf_change:      Callable = None,   # callable(float)
        on_model_change:     Callable = None,   # callable(model_name: str)
        on_box_select:       Callable = None,   # callable(box_index_or_None)
    ):
        super().__init__(master, bg=BG_PANEL, width=280)
        self.pack_propagate(False)

        self._on_yolo        = on_yolo_click
        self._on_yolo_all    = on_yolo_all_click
        self._on_save        = on_save_click
        self._on_clear       = on_clear_click
        self._on_delete_box  = on_delete_box
        self._on_conf_change = on_conf_change
        self._on_model_change = on_model_change
        self._on_box_select  = on_box_select
        self._syncing_selection = False

        # Current class names from YOLO model
        self._class_names: dict[int, str] = {}

        # Manual annotation state
        self.selected_class_var = tk.StringVar(value="dog")
        self.custom_class_var   = tk.StringVar(value="")

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        tk.Label(
            self, text="ANNOTATION PANEL",
            bg=BG_PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"),
        ).pack(pady=(10, 4))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Dark.TNotebook",
            background=BG_PANEL, borderwidth=0,
        )
        style.configure(
            "Dark.TNotebook.Tab",
            background=BG_DARK, foreground=TEXT_LIGHT,
            font=("Consolas", 9, "bold"),
            padding=[10, 4],
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", ACCENT)],
            foreground=[("selected", "white")],
        )

        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        auto_tab   = tk.Frame(nb, bg=BG_PANEL)
        manual_tab = tk.Frame(nb, bg=BG_PANEL)

        nb.add(auto_tab,   text="⚡ Auto (Detector)")
        nb.add(manual_tab, text="✏ Manual")

        self._build_auto_tab(auto_tab)
        self._build_manual_tab(manual_tab)

        # ── shared box list ───────────────────────────────────────────────────
        self._build_box_list()

        # ── bottom buttons ────────────────────────────────────────────────────
        self._build_bottom_buttons()

    def _build_auto_tab(self, parent):
        tk.Label(
            parent, text="Model",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(pady=(10, 2), padx=10, anchor=tk.W)

        tk.Label(
            parent, text=".pt = Ultralytics · .onnx = AGPL-free",
            bg=BG_PANEL, fg="#666688", font=("Consolas", 7),
        ).pack(padx=10, anchor=tk.W)

        model_row = tk.Frame(parent, bg=BG_PANEL)
        model_row.pack(fill=tk.X, padx=10, pady=(4, 6))

        self.model_var = tk.StringVar(value=YOLO_DEFAULT_MODEL)
        self._model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var,
            values=YOLO_MODELS, font=("Consolas", 9), state="readonly", width=14,
        )
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._model_combo.bind("<<ComboboxSelected>>",
                               lambda _e: self._on_model_selected())

        browse_btn = tk.Button(
            model_row, text="📂",
            command=self._browse_model,
            bg=BG_DARK, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=6, font=("Consolas", 10), cursor="hand2",
            activebackground=ACCENT, activeforeground="white", bd=0,
        )
        browse_btn.pack(side=tk.LEFT, padx=(4, 0))
        _hover_btn(browse_btn, BG_DARK, ACCENT)

        conf_hdr = tk.Frame(parent, bg=BG_PANEL)
        conf_hdr.pack(fill=tk.X, padx=10, pady=(8, 2))

        tk.Label(
            conf_hdr, text="Confidence Threshold",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(side=tk.LEFT)

        self.conf_label = tk.Label(
            conf_hdr, text="0.45",
            bg=BG_PANEL, fg=ACCENT, font=("Consolas", 9, "bold"), width=5,
        )
        self.conf_label.pack(side=tk.RIGHT)

        self.conf_var = tk.DoubleVar(value=0.45)
        tk.Scale(
            parent, from_=0.1, to=0.95, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.conf_var,
            command=self._on_conf_slider,
            bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
            highlightthickness=0, sliderrelief=tk.FLAT, showvalue=False,
        ).pack(fill=tk.X, padx=10)

        tk.Label(
            parent, text="Filter Classes (comma-sep, blank=all)",
            bg=BG_PANEL, fg="#666688", font=("Consolas", 7),
        ).pack(pady=(8, 2), padx=10, anchor=tk.W)

        self.filter_var = tk.StringVar(value="")
        tk.Entry(
            parent, textvariable=self.filter_var,
            bg=BG_DARK, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        ).pack(fill=tk.X, padx=10, ipady=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=10, pady=(10, 6))

        yolo_one = tk.Button(
            parent, text="⚡  YOLO This Frame",
            command=self._on_yolo,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=8, pady=7, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#9d8fff", activeforeground="white", bd=0,
        )
        yolo_one.pack(fill=tk.X, padx=10, pady=(0, 3))
        _hover_btn(yolo_one, ACCENT, "#9d8fff")

        yolo_all = tk.Button(
            parent, text="🔁  YOLO All Frames",
            command=self._on_yolo_all,
            bg="#5a4fbf", fg="white", relief=tk.FLAT,
            padx=8, pady=7, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#7a6adf", activeforeground="white", bd=0,
        )
        yolo_all.pack(fill=tk.X, padx=10, pady=(0, 8))
        _hover_btn(yolo_all, "#5a4fbf", "#7a6adf")

    def _build_manual_tab(self, parent):
        tk.Label(
            parent, text="Class for New Box",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(pady=(10, 2), padx=10, anchor=tk.W)

        self.class_combo = ttk.Combobox(
            parent, textvariable=self.selected_class_var,
            font=("Consolas", 9), state="readonly",
        )
        self.class_combo.pack(fill=tk.X, padx=10, ipady=3)

        tk.Label(
            parent, text="or type custom class",
            bg=BG_PANEL, fg="#555577", font=("Consolas", 7),
        ).pack(pady=(8, 2), padx=10, anchor=tk.W)

        tk.Entry(
            parent, textvariable=self.custom_class_var,
            bg=BG_DARK, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        ).pack(fill=tk.X, padx=10, ipady=4)

        tips_frame = tk.Frame(parent, bg=BG_PANEL)
        tips_frame.pack(fill=tk.X, padx=10, pady=(10, 4))

        self._tips_open = False
        self._tips_toggle_btn = tk.Button(
            tips_frame, text="ℹ  How to annotate  ▸",
            bg=BG_DARK, fg="#8888aa", relief=tk.FLAT,
            font=("Consolas", 8), cursor="hand2",
            activebackground=BG_PANEL, activeforeground=TEXT_LIGHT, bd=0,
            anchor=tk.W, command=self._toggle_tips,
        )
        self._tips_toggle_btn.pack(fill=tk.X, ipady=2)

        self._tips_text = tk.Label(
            parent,
            text=(
                "  1. Click '✏ Draw Box' in mode bar\n"
                "  2. Click & drag on the canvas\n"
                "  3. Box is added automatically\n"
                "  4. Navigate frames & repeat\n"
                "  5. Save when done"
            ),
            bg=BG_PANEL, fg="#8888aa",
            font=("Consolas", 8), justify=tk.LEFT,
        )

    def _toggle_tips(self):
        self._tips_open = not self._tips_open
        if self._tips_open:
            self._tips_text.pack(anchor=tk.W, padx=10, pady=(0, 4))
            self._tips_toggle_btn.config(text="ℹ  How to annotate  ▾")
        else:
            self._tips_text.pack_forget()
            self._tips_toggle_btn.config(text="ℹ  How to annotate  ▸")

    def _build_box_list(self):
        sep_f = tk.Frame(self, bg=BG_PANEL)
        sep_f.pack(fill=tk.X, padx=8, pady=(4, 0))
        ttk.Separator(sep_f, orient=tk.HORIZONTAL).pack(fill=tk.X)

        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=8, pady=(4, 2))

        tk.Label(
            hdr, text="DETECTED BOXES",
            bg=BG_PANEL, fg="#888899", font=("Consolas", 7, "bold"),
        ).pack(side=tk.LEFT)

        self.stats_var = tk.StringVar(value="0 boxes")
        tk.Label(
            hdr, textvariable=self.stats_var,
            bg=BG_PANEL, fg=ACCENT, font=("Consolas", 7, "bold"),
        ).pack(side=tk.RIGHT)

        list_frame = tk.Frame(self, bg=BG_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 8), relief=tk.FLAT, bd=0, height=8,
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        scrollbar.config(command=self.listbox.yview)

        del_row = tk.Frame(self, bg=BG_PANEL)
        del_row.pack(fill=tk.X, padx=8, pady=(3, 0))

        del_btn = tk.Button(
            del_row, text="🗑  Delete Selected",
            command=self._delete_selected,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=6, pady=3, font=("Consolas", 8), cursor="hand2",
            activebackground="#a04040", activeforeground="white", bd=0,
        )
        del_btn.pack(side=tk.RIGHT)
        _hover_btn(del_btn, "#7a3333", "#a04040")

    def _build_bottom_buttons(self):
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=8, pady=(6, 4))

        save_btn = tk.Button(
            self, text="💾  Save Annotations",
            command=self._on_save,
            bg="#2d8a4e", fg="white", relief=tk.FLAT,
            padx=8, pady=7, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#3da060", activeforeground="white", bd=0,
        )
        save_btn.pack(fill=tk.X, padx=8, pady=2)
        _hover_btn(save_btn, "#2d8a4e", "#3da060")

        clear_btn = tk.Button(
            self, text="🗑  Clear Frame",
            command=self._on_clear,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=8, pady=7, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#a04040", activeforeground="white", bd=0,
        )
        clear_btn.pack(fill=tk.X, padx=8, pady=(2, 8))
        _hover_btn(clear_btn, "#7a3333", "#a04040")

    # ── model callbacks ───────────────────────────────────────────────────────
    def _on_model_selected(self):
        if self._on_model_change:
            self._on_model_change(self.model_var.get())

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model weights",
            filetypes=[
                ("Model weights", "*.pt *.onnx"),
                ("PyTorch (Ultralytics/AGPL)", "*.pt"),
                ("ONNX (AGPL-free)", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.model_var.set(path)
            current = list(self._model_combo["values"])
            if path not in current:
                self._model_combo["values"] = [path] + current
            if self._on_model_change:
                self._on_model_change(path)

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_conf_slider(self, val):
        fval = float(val)
        if fval >= 0.70:
            color = "#55cc77"
        elif fval >= 0.40:
            color = "#f0c040"
        else:
            color = "#cc5555"
        self.conf_label.config(text=f"{fval:.2f}", fg=color)
        if self._on_conf_change:
            self._on_conf_change(fval)

    def _delete_selected(self):
        sel = self.listbox.curselection()
        if sel and self._on_delete_box:
            self._on_delete_box(sel[0])

    def _on_listbox_select(self, _event):
        if self._syncing_selection:
            return
        sel = self.listbox.curselection()
        idx = sel[0] if sel else None
        if self._on_box_select:
            self._on_box_select(idx)

    def set_selected_box(self, idx):
        """Sync listbox to match canvas selection (without re-firing callback)."""
        self._syncing_selection = True
        try:
            self.listbox.selection_clear(0, tk.END)
            if idx is not None and 0 <= idx < self.listbox.size():
                self.listbox.selection_set(idx)
                self.listbox.see(idx)
        finally:
            self._syncing_selection = False

    # ── public API ────────────────────────────────────────────────────────────
    def update_boxes(self, boxes: list[BoundingBox], class_names: dict[int, str]):
        self._class_names = class_names

        names = sorted(set(class_names.values())) if class_names else ["object"]
        self.class_combo["values"] = names
        if names and self.selected_class_var.get() not in names:
            self.selected_class_var.set(names[0])

        self.listbox.delete(0, tk.END)
        for i, box in enumerate(boxes):
            src  = "YOLO" if box.confidence < 1.0 else " MAN"
            conf = f"{box.confidence:.2f}" if box.confidence < 1.0 else "  — "
            self.listbox.insert(
                tk.END,
                f"  [{i:02d}] {src}  {box.class_name:<14} {conf}",
            )
            row_fg = ACCENT if box.confidence < 1.0 else "#55cc77"
            self.listbox.itemconfig(i, fg=row_fg)

        n = len(boxes)
        self.stats_var.set(f"{n} box{'es' if n != 1 else ''}")

    def get_selected_class(self) -> str:
        """Return custom class if typed, otherwise combo selection."""
        custom = self.custom_class_var.get().strip()
        return custom if custom else self.selected_class_var.get()

    def get_confidence_threshold(self) -> float:
        return float(self.conf_var.get())

    def get_model_name(self) -> str:
        return self.model_var.get()

    def get_class_filter(self) -> list[str]:
        """Return list of class names to keep, or [] for all."""
        raw = self.filter_var.get().strip()
        if not raw:
            return []
        return [c.strip().lower() for c in raw.split(",") if c.strip()]
