"""
ui/export_dialog.py
────────────────────
Modal dialog — pick export format (YOLO Split / YOLO Standard / COCO),
configure Train / Validation / Test split ratios, seed settings, and output folder.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk

from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT


class ExportDialog(tk.Toplevel):
    """
    Export modal dialog with dataset splitting preview and seed options.

    Result dict after closing:
        {
            "format": "yolo_split" | "yolo" | "coco",
            "output_dir": "...",
            "split": bool,
            "train_ratio": float,
            "val_ratio": float,
            "test_ratio": float,
            "seed": int,
            "use_random_seed": bool,
        }
    or None if cancelled.
    """

    def __init__(self, master, default_dir: str = "", total_annotated: int = 100):
        super().__init__(master)
        self.title("Export Dataset")
        self.configure(bg=BG_DARK)
        self.geometry("580x640")
        self.minsize(540, 580)
        self.grab_set()
        self.focus_set()

        self.total_annotated = max(0, total_annotated)
        self.result: dict | None = None

        self.format_var    = tk.StringVar(value="yolo_split")
        self.dir_var       = tk.StringVar(value=default_dir)

        # Split controls
        self.train_pct_var = tk.DoubleVar(value=70.0)
        self.val_pct_var   = tk.DoubleVar(value=20.0)
        self.test_pct_var  = tk.DoubleVar(value=10.0)

        # Seed controls
        self.seed_var      = tk.StringVar(value="42")
        self.rand_seed_var = tk.BooleanVar(value=False)

        # Live status
        self.split_preview_var = tk.StringVar()

        self._build()
        self._center_window(master)
        self._update_split_preview()
        self.wait_window()

    def _center_window(self, master):
        self.update_idletasks()
        try:
            mw = master.winfo_rootx() + master.winfo_width() // 2
            mh = master.winfo_rooty() + master.winfo_height() // 2
            w = self.winfo_width() or 580
            h = self.winfo_height() or 640
            self.geometry(f"{w}x{h}+{max(0, mw - w // 2)}+{max(0, mh - h // 2)}")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill=tk.X, padx=20, pady=(12, 0))

        tk.Label(
            hdr, text="Export Dataset",
            bg=BG_DARK, fg=ACCENT,
            font=("Consolas", 13, "bold"),
        ).pack(anchor=tk.W)

        tk.Label(
            hdr, text=f"Total annotated images available: {self.total_annotated}",
            bg=BG_DARK, fg="#888899", font=("Consolas", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=16, pady=8)

        # Main content area
        main_content = tk.Frame(self, bg=BG_DARK)
        main_content.pack(fill=tk.BOTH, expand=True, padx=20)

        # 1. Format section
        tk.Label(
            main_content, text="Export Format",
            bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 9, "bold"),
        ).pack(anchor=tk.W, pady=(0, 2))

        fmt_box = tk.Frame(main_content, bg=BG_DARK)
        fmt_box.pack(fill=tk.X, pady=(0, 6))

        for value, label, sub in [
            ("yolo_split", "YOLO Dataset Split (Train / Val / Test)",
             "train/, val/, test/ subfolders with images/ & labels/ + data.yaml"),
            ("yolo", "YOLO Standard",
             "Flat images/ + labels/ + data.yaml + classes.txt"),
            ("coco", "COCO JSON",
             "images/ + annotations.json"),
        ]:
            row = tk.Frame(fmt_box, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=2)

            rb = tk.Radiobutton(
                row, text="", variable=self.format_var, value=value,
                bg=BG_PANEL, activebackground=BG_PANEL, selectcolor=ACCENT,
                command=self._on_format_changed,
            )
            rb.pack(side=tk.LEFT, padx=(6, 2))

            txt = tk.Frame(row, bg=BG_PANEL)
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3)
            tk.Label(
                txt, text=label, bg=BG_PANEL, fg=TEXT_LIGHT,
                font=("Consolas", 9, "bold"),
            ).pack(anchor=tk.W)
            tk.Label(
                txt, text=sub, bg=BG_PANEL, fg="#888899",
                font=("Consolas", 8),
            ).pack(anchor=tk.W)

            for w in (row, txt):
                w.bind("<Button-1>", lambda _e, v=value: [self.format_var.set(v), self._on_format_changed()])

        # 2. Dataset Split Section Container
        self.split_container = tk.Frame(main_content, bg=BG_DARK)
        self.split_container.pack(fill=tk.X, pady=4)

        self.split_frame = tk.LabelFrame(
            self.split_container, text=" Dataset Split Options ",
            bg=BG_DARK, fg=ACCENT, font=("Consolas", 9, "bold"),
            padx=10, pady=6,
        )
        self.split_frame.pack(fill=tk.X)

        pct_row = tk.Frame(self.split_frame, bg=BG_DARK)
        pct_row.pack(fill=tk.X, pady=2)

        for label_text, var in [("Train %:", self.train_pct_var), ("Val %:", self.val_pct_var), ("Test %:", self.test_pct_var)]:
            box = tk.Frame(pct_row, bg=BG_DARK)
            box.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
            tk.Label(box, text=label_text, bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 8)).pack(anchor=tk.W)
            sp = tk.Spinbox(
                box, from_=0, to=100, increment=5, textvariable=var,
                width=6, bg=BG_PANEL, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
                relief=tk.FLAT, font=("Consolas", 9),
                command=self._on_ratio_changed,
            )
            sp.pack(anchor=tk.W, ipady=2)
            sp.bind("<KeyRelease>", lambda _e: self._on_ratio_changed())

        # Seed controls
        seed_row = tk.Frame(self.split_frame, bg=BG_DARK)
        seed_row.pack(fill=tk.X, pady=(4, 2))

        tk.Label(seed_row, text="Seed:", bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 8)).pack(side=tk.LEFT, padx=(2, 2))
        self.seed_entry = tk.Entry(
            seed_row, textvariable=self.seed_var, width=8,
            bg=BG_PANEL, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        )
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=2)

        cb = tk.Checkbutton(
            seed_row, text="Random Seed", variable=self.rand_seed_var,
            bg=BG_DARK, fg=TEXT_LIGHT, activebackground=BG_DARK, selectcolor=BG_PANEL,
            font=("Consolas", 8), command=self._on_rand_seed_toggle,
        )
        cb.pack(side=tk.LEFT)

        # Split Live Preview
        preview_lbl = tk.Label(
            self.split_frame, textvariable=self.split_preview_var,
            bg="#111122", fg="#55cc77", font=("Consolas", 8, "bold"),
            padx=8, pady=4, relief=tk.FLAT, justify=tk.LEFT,
        )
        preview_lbl.pack(fill=tk.X, pady=(4, 2))

        # 3. Output folder
        out_box = tk.Frame(main_content, bg=BG_DARK)
        out_box.pack(fill=tk.X, pady=(6, 0))

        tk.Label(
            out_box, text="Output Folder",
            bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 9, "bold"),
        ).pack(anchor=tk.W, pady=(0, 2))

        dir_row = tk.Frame(out_box, bg=BG_DARK)
        dir_row.pack(fill=tk.X)

        tk.Entry(
            dir_row, textvariable=self.dir_var,
            bg=BG_PANEL, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

        tk.Button(
            dir_row, text="Browse…", command=self._browse,
            bg=BG_PANEL, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=10, font=("Consolas", 8, "bold"), cursor="hand2",
            activebackground=ACCENT, activeforeground="white", bd=0,
        ).pack(side=tk.LEFT, ipady=2)

        # 4. Bottom action bar
        footer = tk.Frame(self, bg=BG_DARK)
        footer.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(4, 14))

        ttk.Separator(footer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        btns = tk.Frame(footer, bg=BG_DARK)
        btns.pack(fill=tk.X)

        tk.Button(
            btns, text="✖  Cancel", command=self.destroy,
            bg="#444455", fg="white", relief=tk.FLAT,
            padx=14, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#555566", activeforeground="white", bd=0,
        ).pack(side=tk.LEFT)

        self.confirm_btn = tk.Button(
            btns, text="🚀  Generate Dataset", command=self._confirm,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=16, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#9d8fff", activeforeground="white", bd=0,
        )
        self.confirm_btn.pack(side=tk.RIGHT)

    # ── callbacks & preview calculation ─────────────────────────────────────

    def _on_format_changed(self):
        is_split = self.format_var.get() == "yolo_split"
        if is_split:
            self.confirm_btn.config(text="🚀  Generate Dataset")
            self.split_container.pack(fill=tk.X, pady=4)
        else:
            self.confirm_btn.config(text="📤  Export")
            self.split_container.pack_forget()

    def _on_rand_seed_toggle(self):
        if self.rand_seed_var.get():
            self.seed_entry.config(state="disabled")
        else:
            self.seed_entry.config(state="normal")

    def _on_ratio_changed(self):
        self._update_split_preview()

    def _update_split_preview(self):
        try:
            tr = float(self.train_pct_var.get())
            va = float(self.val_pct_var.get())
            te = float(self.test_pct_var.get())
        except (ValueError, tk.TclError):
            tr, va, te = 70.0, 20.0, 10.0

        total_pct = tr + va + te
        if total_pct <= 0:
            tr, va, te = 70.0, 20.0, 10.0
            total_pct = 100.0

        r_tr, r_va, r_te = tr / total_pct, va / total_pct, te / total_pct

        total = self.total_annotated
        n_tr = int(round(total * r_tr))
        n_va = int(round(total * r_va))
        n_te = max(0, total - n_tr - n_va)

        pct_tr = (n_tr / total * 100.0) if total > 0 else tr
        pct_va = (n_va / total * 100.0) if total > 0 else va
        pct_te = (n_te / total * 100.0) if total > 0 else te

        msg = (
            f"Train: {n_tr} ({pct_tr:.1f}%)   |   "
            f"Val: {n_va} ({pct_va:.1f}%)   |   "
            f"Test: {n_te} ({pct_te:.1f}%)\n"
            f"Total: {total} annotated image(s) split across sets"
        )
        self.split_preview_var.set(msg)

    def _browse(self):
        path = filedialog.askdirectory(
            title="Choose Export Folder", parent=self,
            initialdir=self.dir_var.get() or os.getcwd(),
        )
        if path:
            self.dir_var.set(path)

    def _confirm(self):
        from tkinter import messagebox
        out = self.dir_var.get().strip()
        if not out:
            messagebox.showwarning(
                "No folder",
                "Please choose an output folder.", parent=self,
            )
            return

        fmt = self.format_var.get()
        try:
            tr = max(0.0, float(self.train_pct_var.get()))
            va = max(0.0, float(self.val_pct_var.get()))
            te = max(0.0, float(self.test_pct_var.get()))
        except ValueError:
            tr, va, te = 70.0, 20.0, 10.0

        tot = tr + va + te
        if tot <= 0:
            tot = 100.0
            tr, va, te = 70.0, 20.0, 10.0

        try:
            seed_val = int(self.seed_var.get().strip()) if self.seed_var.get().strip() else 42
        except ValueError:
            seed_val = 42

        self.result = {
            "format":          fmt,
            "output_dir":      out,
            "split":           (fmt == "yolo_split"),
            "train_ratio":     tr / tot,
            "val_ratio":       va / tot,
            "test_ratio":      te / tot,
            "seed":            seed_val,
            "use_random_seed": self.rand_seed_var.get(),
        }
        self.destroy()
