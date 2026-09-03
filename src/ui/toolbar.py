"""
ui/toolbar.py
──────────────
Top toolbar with file-open, mode-toggle, save, and Annotate buttons.
"""
import tkinter as tk

from utils.config import ACCENT, BG_PANEL, TEXT_LIGHT


class Toolbar(tk.Frame):
    """
    Horizontal toolbar extracted from main_window.

    Parameters
    ----------
    master    : parent widget
    callbacks : dict[str, callable] with keys:
                  on_open_video, on_open_images, on_open_folder,
                  on_save, on_clear, on_draw_mode, on_view_mode, on_yolo_all
    """

    _BG         = BG_PANEL
    _BTN_BG     = "#333355"
    _BTN_FG     = TEXT_LIGHT
    _ACTIVE_BG  = ACCENT
    _SEP_COLOR  = "#3a3a5e"

    def __init__(self, master: tk.Misc, callbacks: dict, **kw) -> None:
        kw.setdefault("bg", self._BG)
        kw.setdefault("height", 36)
        super().__init__(master, **kw)
        self.pack_propagate(False)

        self._callbacks = callbacks
        self._mode_btns: dict[str, tk.Button] = {}

        # ── Open section ──────────────────────────────────────────────────────
        self._btn("Open Video",   callbacks.get("on_open_video"))
        self._btn("Open Images",  callbacks.get("on_open_images"))
        self._btn("Open Folder",  callbacks.get("on_open_folder"))
        self._sep()

        # ── Mode section ──────────────────────────────────────────────────────
        self._mode_btns["VIEW"] = self._btn(
            "VIEW", callbacks.get("on_view_mode"), tag="VIEW"
        )
        self._mode_btns["DRAW"] = self._btn(
            "DRAW", callbacks.get("on_draw_mode"), tag="DRAW"
        )
        self._sep()

        # ── Actions ───────────────────────────────────────────────────────────
        self._btn("Save",      callbacks.get("on_save"))
        self._btn("Clear",     callbacks.get("on_clear"))
        self._btn("Annotate All", callbacks.get("on_yolo_all"))

        # Default mode highlight
        self.set_mode("VIEW")

    # ── public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Highlight the active mode button (VIEW or DRAW)."""
        for name, btn in self._mode_btns.items():
            if name == mode.upper():
                btn.configure(bg=ACCENT, fg=TEXT_LIGHT, relief=tk.SUNKEN)
            else:
                btn.configure(bg=self._BTN_BG, fg=self._BTN_FG, relief=tk.FLAT)

    # ── private ───────────────────────────────────────────────────────────────

    def _btn(
        self,
        label: str,
        command,
        tag: str = "",
    ) -> tk.Button:
        btn = tk.Button(
            self,
            text=label,
            command=command or (lambda: None),
            bg=self._BTN_BG,
            fg=self._BTN_FG,
            activebackground=ACCENT,
            activeforeground=TEXT_LIGHT,
            relief=tk.FLAT,
            cursor="hand2",
            padx=8,
            pady=3,
            bd=0,
            font=("Helvetica", 9),
        )
        btn.pack(side=tk.LEFT, padx=2, pady=3)
        return btn

    def _sep(self) -> None:
        sep = tk.Frame(self, bg=self._SEP_COLOR, width=1)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=4)
