"""
ui/quality_report_dialog.py
────────────────────────────
Interactive modal dialog showing annotation quality check results.

Displays a summary (✅ valid / ⚠️ warnings / ❌ errors) and a scrollable
list of issues.  Clicking an issue invokes a callback to navigate to the
problematic frame/annotation.
"""
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from core.quality_checker import QualityIssue, QualityReport, Severity
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT

# ── Colour tokens ────────────────────────────────────────────────────────────
_CLR_ERROR   = "#e05c5c"
_CLR_WARN    = "#e0a830"
_CLR_VALID   = "#55cc77"
_CLR_HOVER   = "#3a3a5e"
_CLR_ROW_BG  = "#252538"
_CLR_ROW_ALT = "#2a2a42"


class QualityReportDialog(tk.Toplevel):
    """
    Modal quality-report window.

    Parameters
    ----------
    master          : parent Tk widget.
    report          : a ``QualityReport`` from the checker engine.
    on_jump         : callback(frame_index, annotation_type, annotation_index)
                      called when the user clicks an issue row.
    on_rerun        : callback()  — called when "Run Quality Check" is pressed.
    """

    def __init__(
        self,
        master,
        report: QualityReport,
        on_jump: Callable | None = None,
        on_rerun: Callable | None = None,
    ):
        super().__init__(master)
        self.title("Annotation Quality Report")
        self.configure(bg=BG_DARK)
        self.geometry("680x620")
        self.minsize(580, 480)
        self.grab_set()
        self.focus_set()

        self.report   = report
        self._on_jump  = on_jump
        self._on_rerun = on_rerun

        self._build()
        self._populate()
        self._center(master)

    # ── centre on parent ─────────────────────────────────────────────────────
    def _center(self, master):
        self.update_idletasks()
        try:
            mw = master.winfo_rootx() + master.winfo_width() // 2
            mh = master.winfo_rooty() + master.winfo_height() // 2
            w = self.winfo_width() or 680
            h = self.winfo_height() or 620
            self.geometry(f"{w}x{h}+{max(0, mw - w // 2)}+{max(0, mh - h // 2)}")
        except Exception:  # nosec B110
            pass

    # ── UI construction ──────────────────────────────────────────────────────
    def _build(self):
        # ─── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill=tk.X, padx=20, pady=(14, 0))

        tk.Label(
            hdr, text="🔍  Annotation Quality Report",
            bg=BG_DARK, fg=ACCENT,
            font=("Consolas", 14, "bold"),
        ).pack(anchor=tk.W)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=16, pady=8)

        # ─── Summary cards ───────────────────────────────────────────────────
        self._summary_frame = tk.Frame(self, bg=BG_DARK)
        self._summary_frame.pack(fill=tk.X, padx=20, pady=(0, 4))

        # ─── Filter bar ─────────────────────────────────────────────────────
        filt = tk.Frame(self, bg=BG_DARK)
        filt.pack(fill=tk.X, padx=20, pady=(0, 4))

        tk.Label(
            filt, text="Filter:", bg=BG_DARK, fg="#888899",
            font=("Consolas", 8),
        ).pack(side=tk.LEFT, padx=(0, 6))

        self._filter_var = tk.StringVar(value="all")
        for value, label in [("all", "All"), ("error", "❌ Errors"), ("warning", "⚠️ Warnings")]:
            rb = tk.Radiobutton(
                filt, text=label, variable=self._filter_var, value=value,
                bg=BG_DARK, fg=TEXT_LIGHT, activebackground=BG_DARK,
                selectcolor=BG_PANEL, font=("Consolas", 8),
                command=self._populate,
            )
            rb.pack(side=tk.LEFT, padx=4)

        # ─── Scrollable issue list ───────────────────────────────────────────
        list_frame = tk.Frame(self, bg=BG_DARK)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 4))

        self._canvas = tk.Canvas(list_frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=BG_DARK)

        self._inner.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Make inner frame fill canvas width
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Mouse-wheel scrolling
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind("<Destroy>", self._on_destroy)

        # ─── Footer buttons ──────────────────────────────────────────────────
        footer = tk.Frame(self, bg=BG_DARK)
        footer.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(4, 14))

        ttk.Separator(footer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        btns = tk.Frame(footer, bg=BG_DARK)
        btns.pack(fill=tk.X)

        tk.Button(
            btns, text="✖  Close", command=self.destroy,
            bg="#444455", fg="white", relief=tk.FLAT,
            padx=14, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#555566", activeforeground="white", bd=0,
        ).pack(side=tk.LEFT)

        if self._on_rerun:
            tk.Button(
                btns, text="🔄  Run Quality Check", command=self._rerun,
                bg="#1f7a8c", fg="white", relief=tk.FLAT,
                padx=14, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
                activebackground="#2a9aae", activeforeground="white", bd=0,
            ).pack(side=tk.RIGHT, padx=(6, 0))

        self._review_btn = tk.Button(
            btns, text="🔎  Fix / Review Issues", command=self._review_first,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=14, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
            activebackground="#9d8fff", activeforeground="white", bd=0,
        )
        self._review_btn.pack(side=tk.RIGHT)

    # ── Summary cards ────────────────────────────────────────────────────────
    def _build_summary(self):
        for w in self._summary_frame.winfo_children():
            w.destroy()

        r = self.report
        valid_total = r.valid_boxes + r.valid_polygons

        cards = [
            ("✅  Valid",    str(valid_total),    _CLR_VALID,  f"{r.valid_boxes} box(es), {r.valid_polygons} polygon(s)"),
            ("⚠️  Warnings", str(r.warning_count), _CLR_WARN,   "Click to filter"),
            ("❌  Errors",   str(r.error_count),   _CLR_ERROR,  "Click to filter"),
        ]

        for title, count, color, sub in cards:
            card = tk.Frame(self._summary_frame, bg=BG_PANEL, padx=12, pady=8)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3, pady=2)

            tk.Label(
                card, text=title, bg=BG_PANEL, fg=color,
                font=("Consolas", 9, "bold"),
            ).pack(anchor=tk.W)

            tk.Label(
                card, text=count, bg=BG_PANEL, fg="white",
                font=("Consolas", 20, "bold"),
            ).pack(anchor=tk.W, pady=(2, 0))

            tk.Label(
                card, text=sub, bg=BG_PANEL, fg="#888899",
                font=("Consolas", 7),
            ).pack(anchor=tk.W)

        # Stats bar
        stats = tk.Frame(self._summary_frame, bg=BG_DARK)
        stats.pack(fill=tk.X, pady=(6, 0))

        info = (
            f"Scanned {r.total_frames} frame(s)  ·  "
            f"{r.annotated_frames} annotated  ·  "
            f"{r.total_boxes} box(es)  ·  "
            f"{r.total_polygons} polygon(s)"
        )
        tk.Label(
            stats, text=info, bg=BG_DARK, fg="#666688",
            font=("Consolas", 8),
        ).pack(anchor=tk.W, padx=4)

    # ── Populate issue rows ──────────────────────────────────────────────────
    def _populate(self, *_args):
        self._build_summary()

        for w in self._inner.winfo_children():
            w.destroy()

        filt = self._filter_var.get()
        issues = self.report.issues
        if filt == "error":
            issues = [i for i in issues if i.severity == Severity.ERROR]
        elif filt == "warning":
            issues = [i for i in issues if i.severity == Severity.WARNING]

        if not issues:
            if self.report.is_clean:
                msg = "🎉  All annotations passed quality checks!"
                color = _CLR_VALID
            else:
                msg = "No issues match the current filter."
                color = "#888899"

            tk.Label(
                self._inner, text=msg,
                bg=BG_DARK, fg=color,
                font=("Consolas", 11, "bold"),
                pady=40,
            ).pack(fill=tk.X)

            self._review_btn.config(state=tk.DISABLED)
            return

        self._review_btn.config(state=tk.NORMAL)

        for idx, issue in enumerate(issues):
            bg = _CLR_ROW_BG if idx % 2 == 0 else _CLR_ROW_ALT
            self._add_issue_row(issue, bg)

    def _add_issue_row(self, issue: QualityIssue, bg: str):
        row = tk.Frame(self._inner, bg=bg, cursor="hand2")
        row.pack(fill=tk.X, pady=1, padx=2)

        # Severity icon
        sev_color = _CLR_ERROR if issue.severity == Severity.ERROR else _CLR_WARN
        tk.Label(
            row, text=issue.icon, bg=bg,
            font=("Consolas", 11), width=3,
        ).pack(side=tk.LEFT, padx=(6, 2), pady=6)

        # Frame badge
        frame_badge = tk.Label(
            row,
            text=f"Frame {issue.frame_index + 1}",
            bg=sev_color, fg="white",
            font=("Consolas", 7, "bold"),
            padx=5, pady=1,
        )
        frame_badge.pack(side=tk.LEFT, padx=(0, 8), pady=6)

        # Message
        msg_lbl = tk.Label(
            row, text=issue.message, bg=bg, fg=TEXT_LIGHT,
            font=("Consolas", 8), anchor=tk.W, justify=tk.LEFT,
            wraplength=420,
        )
        msg_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), pady=6)

        # Jump arrow
        arrow = tk.Label(
            row, text="→", bg=bg, fg="#666688",
            font=("Consolas", 12, "bold"),
        )
        arrow.pack(side=tk.RIGHT, padx=(0, 10), pady=6)

        # Click binding on every widget in the row
        def _on_click(_e, iss=issue):
            self._jump_to(iss)

        for widget in (row, frame_badge, msg_lbl, arrow):
            widget.bind("<Button-1>", _on_click)

        # Hover effect
        def _enter(_e, r=row, widgets=(row, msg_lbl, arrow)):
            for w in widgets:
                w.config(bg=_CLR_HOVER)

        def _leave(_e, r=row, orig=bg, widgets=(row, msg_lbl, arrow)):
            for w in widgets:
                w.config(bg=orig)

        for widget in (row, msg_lbl, arrow):
            widget.bind("<Enter>", _enter)
            widget.bind("<Leave>", _leave)

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _jump_to(self, issue: QualityIssue):
        if self._on_jump:
            self._on_jump(issue.frame_index, issue.annotation_type, issue.annotation_index)

    def _review_first(self):
        """Jump to the first error; if none, jump to the first warning."""
        target = None
        for issue in self.report.issues:
            if issue.severity == Severity.ERROR:
                target = issue
                break
        if target is None:
            for issue in self.report.issues:
                if issue.severity == Severity.WARNING:
                    target = issue
                    break
        if target:
            self._jump_to(target)

    def _rerun(self):
        if self._on_rerun:
            self._on_rerun()

    def update_report(self, report: QualityReport):
        """Replace the current report and refresh the display."""
        self.report = report
        self._populate()

    # ── Scrolling helpers ────────────────────────────────────────────────────
    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_destroy(self, _event):
        try:
            self._canvas.unbind_all("<MouseWheel>")
        except Exception:  # nosec B110
            pass
