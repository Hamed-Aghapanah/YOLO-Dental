#!/usr/bin/env python
"""
opg_dashboard.py
================
Themed, animated CustomTkinter dashboard for the OPG third-molar project.

    pip install customtkinter ultralytics opencv-python pandas matplotlib openpyxl pillow
    python opg_dashboard.py

Tabs
    1  Data        class A/B/C composition across train/val/test, charts, Excel export
    2  Metrics     every model x task, measured or manuscript-sourced, Excel export
    3  Confusion   per model/task matrix, numbers / percent / both, Excel export
    4  Test        run a .pt on an image: heat map, mask, contour, confidence overlays
    5  Proposed    what the numbers say and what to do next
    6  Requirements  environment check with copyable fix commands

A NOTE ON PROVENANCE, BECAUSE IT DECIDES WHAT YOU MAY PUBLISH
------------------------------------------------------------
Where trained weights exist, every number here is measured from cached predictions
on the held-out test split.  Where they do not, the dashboard falls back to the
numbers already written in the manuscript so you can see the shape of the final
table early.  Those rows are tagged "manuscript", coloured amber, and exported in a
separate column.  They are the paper's own numbers being shown back to you; copying
them into the paper as results would be circular.  The status bar states which mode
each tab is in.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
except ImportError:  # pragma: no cover
    raise SystemExit("customtkinter is missing.  pip install customtkinter")

import cv2
import numpy as np
import pandas as pd

import opg_scan as SCAN
import opg_viz as VIZ

# --------------------------------------------------------------------------------------
# themes
# --------------------------------------------------------------------------------------
THEMES = {
    "Midnight": dict(mode="dark", accent="#4c9be8", accent2="#7fd4d4",
                     bg="#161821", panel="#1e2130", card="#252a3a",
                     text="#e6e9f0", sub="#98a0b3", ok="#59c17a",
                     warn="#e8a24c", bad="#e05f5f", cmap=cv2.COLORMAP_JET,
                     mpl="#1e2130"),
    "Clinical": dict(mode="light", accent="#1f6feb", accent2="#0a7ea4",
                     bg="#f4f6fa", panel="#ffffff", card="#eef2f8",
                     text="#1b2330", sub="#5b6676", ok="#1a7f4b",
                     warn="#b26a00", bad="#c0392b", cmap=cv2.COLORMAP_VIRIDIS,
                     mpl="#ffffff"),
    "Amber": dict(mode="dark", accent="#e8a24c", accent2="#ffd479",
                  bg="#1a1713", panel="#241f19", card="#2f2921",
                  text="#f0e9df", sub="#b3a48f", ok="#8fbf5f",
                  warn="#e8c14c", bad="#e07a5f", cmap=cv2.COLORMAP_INFERNO,
                  mpl="#241f19"),
    "Teal": dict(mode="dark", accent="#2ec4b6", accent2="#8ef0e6",
                 bg="#101b1c", panel="#16262a", card="#1d3238",
                 text="#e3f2f1", sub="#93b3b3", ok="#5fd68a",
                 warn="#e8c14c", bad="#e06a6a", cmap=cv2.COLORMAP_TURBO,
                 mpl="#16262a"),
    "Slate": dict(mode="dark", accent="#8ea3c4", accent2="#c3d0e3",
                  bg="#15171c", panel="#1d2027", card="#262a33",
                  text="#e4e7ee", sub="#98a0b0", ok="#6fbf8a",
                  warn="#d6a55c", bad="#d67a7a", cmap=cv2.COLORMAP_BONE,
                  mpl="#1d2027"),
}


# --------------------------------------------------------------------------------------
# animation helpers
# --------------------------------------------------------------------------------------
def ease_out_cubic(t: float) -> float:
    return 1 - pow(1 - t, 3)


def ease_in_out(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in c)


def lerp_color(a, b, t):
    ca, cb = _hex_to_rgb(a), _hex_to_rgb(b)
    return _rgb_to_hex([ca[i] + (cb[i] - ca[i]) * t for i in range(3)])


class Animator:
    """after()-driven tweens. Every animation registers so it can be cancelled."""

    def __init__(self, widget):
        self.w = widget
        self._jobs = []
        self.alive = True



    def tween(self, ms, step_fn, done_fn=None, fps=60, ease=ease_out_cubic):
        n = max(1, int(ms / (1000 / fps)))
        state = {"i": 0}

        def tick():
            if not self.alive:
                return
            state["i"] += 1
            t = min(1.0, state["i"] / n)
            try:
                step_fn(ease(t))
            except Exception:  # noqa: BLE001
                return
            if t < 1.0:
                job = self.w.after(int(1000 / fps), tick)
                self._jobs.append(job)
            elif done_fn:
                try:
                    done_fn()
                except Exception:  # noqa: BLE001
                    pass

        if self.alive:
            tick()

    def count_up(self, label, target, ms=700, fmt="{:,.0f}", prefix="", suffix=""):
        try:
            target = float(target)
        except (TypeError, ValueError):
            label.configure(text=f"{prefix}{target}{suffix}")
            return
        self.tween(ms, lambda t: label.configure(
            text=f"{prefix}{fmt.format(target * t)}{suffix}"))

    def fade_text(self, label, frm, to, ms=400):
        self.tween(ms, lambda t: label.configure(text_color=lerp_color(frm, to, t)))

    def slide_in(self, widget, ms=350, dy=18):
        def step(t):
            try:
                widget.grid_configure(pady=(int(dy * (1 - t)), 0))
            except Exception:  # noqa: BLE001
                pass
        self.tween(ms, step)

    def pulse(self, widget, c1, c2, period=1100):
        state = {"up": True}

        def loop():
            frm, to = (c1, c2) if state["up"] else (c2, c1)
            state["up"] = not state["up"]
            self.tween(period, lambda t: widget.configure(
                fg_color=lerp_color(frm, to, t)),
                done_fn=lambda: self._jobs.append(self.w.after(20, loop)))
        loop()

    def cancel_all(self):
        """Drop pending tweens. Used on theme change; animation stays enabled."""
        for j in self._jobs:
            try:
                self.w.after_cancel(j)
            except Exception:  # noqa: BLE001
                pass
        self._jobs.clear()

    def shutdown(self):
        """Cancel everything AND disable further scheduling.

        Without this, tweens keep calling after() while the window is being torn
        down and Tk reports 'invalid command name ...tick' / '...update' once the
        widget behind the callback no longer exists - exactly the errors seen on
        exit in the console log.
        """
        self.alive = False
        self.cancel_all()


# --------------------------------------------------------------------------------------
class App(ctk.CTk):
    """
    Standalone window.  `App(master=...)` is not used; to embed the analysis tabs in
    another application, launch DashboardWindow (below), which is a CTkToplevel and
    shares the parent's event loop.
    """
    def __init__(self):
        super().__init__()
        self.theme_name = "Midnight"
        self.T = THEMES[self.theme_name]
        ctk.set_appearance_mode(self.T["mode"])
        ctk.set_default_color_theme("blue")

        self.title("OPG Third-Molar Dashboard")
        self.geometry("1620x1000")
        self.minsize(1280, 820)
        self.configure(fg_color=self.T["bg"])
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.anim = Animator(self)
        self.q: queue.Queue = queue.Queue()
        self.disc = SCAN.Discovery()
        self.metrics_df = pd.DataFrame()
        self.comp = {}
        self.predictor = None
        self.current_image = None
        self.current_render = None

        self._alive = True
        self._jobs = []
        self._build_header()
        self._build_tabs()
        self._jobs.append(self.after(80, self._drain))
        self._jobs.append(self.after(250, self._intro))

    # ------------------------------------------------------------------ shutdown
    def _shutdown(self):
        """Cancel every pending after() callback before the widgets disappear."""
        self._alive = False
        try:
            self.anim.shutdown()
        except Exception:  # noqa: BLE001
            pass
        for j in list(self._jobs):
            try:
                self.after_cancel(j)
            except Exception:  # noqa: BLE001
                pass
        self._jobs = []
        for attr in ("_conf_job",):
            j = getattr(self, attr, None)
            if j:
                try:
                    self.after_cancel(j)
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)

    def _on_close(self):
        self._shutdown()
        try:
            self.quit()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()

    # ============================================================ header / theme
    def _build_header(self):
        h = ctk.CTkFrame(self, fg_color=self.T["panel"], corner_radius=0, height=76)
        h.grid(row=0, column=0, sticky="ew")
        h.grid_columnconfigure(3, weight=1)
        self.header = h

        self.dot = ctk.CTkFrame(h, width=14, height=14, corner_radius=7,
                                fg_color=self.T["accent"])
        self.dot.grid(row=0, column=0, padx=(18, 10), pady=24)

        self.title_lbl = ctk.CTkLabel(h, text="OPG Third-Molar Dashboard",
                                      font=ctk.CTkFont(size=21, weight="bold"),
                                      text_color=self.T["text"])
        self.title_lbl.grid(row=0, column=1, sticky="w")
        self.sub_lbl = ctk.CTkLabel(h, text="detection · grading · reporting",
                                    font=ctk.CTkFont(size=12),
                                    text_color=self.T["sub"])
        self.sub_lbl.grid(row=0, column=2, padx=14, sticky="w")

        right = ctk.CTkFrame(h, fg_color="transparent")
        right.grid(row=0, column=4, padx=16, sticky="e")
        ctk.CTkLabel(right, text="Theme", text_color=self.T["sub"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
        self.theme_menu = ctk.CTkOptionMenu(right, values=list(THEMES), width=130,
                                            command=self._set_theme)
        self.theme_menu.set(self.theme_name)
        self.theme_menu.pack(side="left", padx=4)
        self.scan_btn = ctk.CTkButton(right, text="Rescan", width=90,
                                      command=self.rescan)
        self.scan_btn.pack(side="left", padx=6)

        self.status = ctk.CTkLabel(self, text="ready", anchor="w",
                                   font=ctk.CTkFont(size=12),
                                   text_color=self.T["sub"])
        self.status.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 8))

    def _intro(self):
        self.anim.pulse(self.dot, self.T["accent"], self.T["panel"])
        self.anim.fade_text(self.title_lbl, self.T["panel"], self.T["text"], 600)

    def _set_theme(self, name):
        self.theme_name = name
        self.T = THEMES[name]
        ctk.set_appearance_mode(self.T["mode"])
        self.anim.cancel_all()
        self.configure(fg_color=self.T["bg"])
        self.header.configure(fg_color=self.T["panel"])
        self.title_lbl.configure(text_color=self.T["text"])
        self.sub_lbl.configure(text_color=self.T["sub"])
        self.status.configure(text_color=self.T["sub"])
        self._restyle_trees()
        self.anim.pulse(self.dot, self.T["accent"], self.T["panel"])
        self.set_status(f"theme: {name}")
        for fn in (self._draw_data_charts, self._draw_confusion):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def set_status(self, msg, kind="info"):
        col = {"info": self.T["sub"], "ok": self.T["ok"],
               "warn": self.T["warn"], "bad": self.T["bad"]}[kind]
        self.status.configure(text=msg)
        self.anim.fade_text(self.status, self.T["panel"], col, 350)

    # ================================================================= tab shell
    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self, fg_color=self.T["panel"],
                                   segmented_button_selected_color=self.T["accent"])
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        for n in ("1 · Data", "2 · Metrics", "3 · Confusion", "4 · Test",
                  "5 · Proposed", "6 · Requirements"):
            self.tabs.add(n)
        self._tab_data()
        self._tab_metrics()
        self._tab_confusion()
        self._tab_test()
        self._tab_proposed()
        self._tab_requirements()
        self._restyle_trees()

    def _tree(self, parent, cols, height=12):
        f = ctk.CTkFrame(parent, fg_color=self.T["card"])
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(0, weight=1)
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:  # noqa: BLE001
            pass
        t = ttk.Treeview(f, columns=cols, show="headings", height=height)
        for c in cols:
            t.heading(c, text=str(c))
            t.column(c, width=max(70, min(210, 9 * len(str(c)) + 52)), anchor="center")
        vs = ttk.Scrollbar(f, orient="vertical", command=t.yview)
        hs = ttk.Scrollbar(f, orient="horizontal", command=t.xview)
        t.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        t.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        return f, t

    def _restyle_trees(self):
        st = ttk.Style()
        st.configure("Treeview", background=self.T["card"], foreground=self.T["text"],
                     fieldbackground=self.T["card"], rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=self.T["panel"],
                     foreground=self.T["accent2"], relief="flat")
        st.map("Treeview", background=[("selected", self.T["accent"])])
        for t in getattr(self, "_all_trees", []):
            try:
                t.tag_configure("manuscript", foreground=self.T["warn"])
                t.tag_configure("measured", foreground=self.T["text"])
                t.tag_configure("total", foreground=self.T["accent2"])
            except Exception:  # noqa: BLE001
                pass

    def _fill(self, tree, df, tag_col="source"):
        for i in tree.get_children():
            tree.delete(i)
        if df is None or len(df) == 0:
            return
        cols = list(tree["columns"])
        for _, r in df.iterrows():
            tag = ""
            if tag_col in df.columns:
                tag = "manuscript" if str(r.get(tag_col)) == "manuscript" else "measured"
            first = str(r.get(cols[0], ""))
            if first.upper() in ("TOTAL", "ALL"):
                tag = "total"
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = "" if pd.isna(v) else round(v, 4)
                vals.append(v)
            tree.insert("", "end", values=vals, tags=(tag,))

    # ================================================================== Tab 1
    def _tab_data(self):
        t = self.tabs.tab("1 · Data")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(3, weight=1)
        t.grid_rowconfigure(4, weight=1)

        top = ctk.CTkFrame(t, fg_color=self.T["card"])
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Project folder", width=110,
                     anchor="w").grid(row=0, column=0, padx=10, pady=8)
        self.root_var = ctk.StringVar(value=r"E:\project\0000_OPG\203-opg")
        ctk.CTkEntry(top, textvariable=self.root_var).grid(row=0, column=1,
                                                           sticky="ew", padx=6)
        ctk.CTkButton(top, text="...", width=40,
                      command=lambda: self._pick_dir(self.root_var)).grid(row=0, column=2,
                                                                          padx=6)
        ctk.CTkLabel(top, text="Output folder", width=110,
                     anchor="w").grid(row=1, column=0, padx=10, pady=(0, 8))
        self.out_var = ctk.StringVar(value=r"E:\project\0000_OPG\203-opg\output")
        ctk.CTkEntry(top, textvariable=self.out_var).grid(row=1, column=1,
                                                          sticky="ew", padx=6)
        ctk.CTkButton(top, text="...", width=40,
                      command=lambda: self._pick_dir(self.out_var)).grid(row=1, column=2,
                                                                         padx=6)
        ctk.CTkButton(top, text="Scan everything", width=150,
                      command=self.rescan).grid(row=0, column=3, rowspan=2, padx=12)

        cards = ctk.CTkFrame(t, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", padx=8)
        self.stat_cards = {}
        for i, (key, label) in enumerate(
                [("images", "Panoramic images"), ("molars", "Graded molars"),
                 ("A", "Class A"), ("B", "Class B"), ("C", "Class C"),
                 ("both", "Both sides")]):
            cards.grid_columnconfigure(i, weight=1)
            c = ctk.CTkFrame(cards, fg_color=self.T["card"], corner_radius=12)
            c.grid(row=0, column=i, sticky="ew", padx=6, pady=6)
            val = ctk.CTkLabel(c, text="0", font=ctk.CTkFont(size=26, weight="bold"),
                               text_color=self.T["accent"])
            val.pack(pady=(14, 2))
            ctk.CTkLabel(c, text=label, font=ctk.CTkFont(size=11),
                         text_color=self.T["sub"]).pack(pady=(0, 12))
            self.stat_cards[key] = val

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 0))
        ctk.CTkButton(bar, text="Export to Excel", width=150,
                      command=self._export_data).pack(side="left", padx=4)
        self.data_src = ctk.CTkLabel(bar, text="", text_color=self.T["sub"],
                                     font=ctk.CTkFont(size=11), anchor="w")
        self.data_src.pack(side="left", padx=14)

        self._all_trees = []
        f, self.data_tree = self._tree(t, ("split", "A", "B", "C",
                                           "total molars", "images"), height=7)
        f.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)
        self._all_trees.append(self.data_tree)

        self.data_chart = ctk.CTkFrame(t, fg_color=self.T["card"])
        self.data_chart.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _draw_data_charts(self):
        for w in self.data_chart.winfo_children():
            w.destroy()
        comp = self.comp
        if not comp or comp.get("by_split") is None or len(comp["by_split"]) == 0:
            ctk.CTkLabel(self.data_chart,
                         text="Scan a project folder to see the composition.",
                         text_color=self.T["sub"]).pack(pady=30)
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:  # noqa: BLE001
            ctk.CTkLabel(self.data_chart, text=f"matplotlib unavailable: {e}").pack()
            return

        bs = comp["by_split"]
        bs = bs[bs["split"] != "TOTAL"]
        tot = comp["totals"]
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.5), dpi=100)
        fig.patch.set_facecolor(self.T["mpl"])
        cols = [self.T["accent"], self.T["accent2"], self.T["warn"]]

        ax = axes[0]
        x = np.arange(len(bs))
        bottom = np.zeros(len(bs))
        for i, c in enumerate(SCAN.CLASS_NAMES):
            v = bs[c].to_numpy(float) if c in bs else np.zeros(len(bs))
            ax.bar(x, v, 0.62, bottom=bottom, label=f"Class {c}", color=cols[i])
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(bs["split"], rotation=12)
        ax.set_title("Molars per split", color=self.T["text"], fontsize=11)
        ax.legend(fontsize=8, facecolor=self.T["mpl"], edgecolor="#555",
                  labelcolor=self.T["text"])

        ax = axes[1]
        vals = tot["molars"].to_numpy(float)
        wedges, *_ = ax.pie(vals, colors=cols, startangle=110,
                            wedgeprops=dict(width=0.42, edgecolor=self.T["mpl"]))
        ax.set_title("Class balance", color=self.T["text"], fontsize=11)
        n = vals.sum()
        ax.text(0, 0, f"{int(n)}\nmolars", ha="center", va="center",
                color=self.T["text"], fontsize=11)
        ax.legend(wedges, [f"{c}  {int(v)}  ({100*v/max(1,n):.1f}%)"
                           for c, v in zip(SCAN.CLASS_NAMES, vals)],
                  fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  facecolor=self.T["mpl"], edgecolor="#555", labelcolor=self.T["text"])

        ax = axes[2]
        side = comp.get("by_side")
        if side is not None and len(side):
            x = np.arange(len(side))
            bottom = np.zeros(len(side))
            for i, c in enumerate(SCAN.CLASS_NAMES):
                v = side[c].to_numpy(float) if c in side else np.zeros(len(side))
                ax.bar(x, v, 0.5, bottom=bottom, color=cols[i])
                bottom += v
            ax.set_xticks(x)
            ax.set_xticklabels([f"patient {s}" for s in side["patient side"]])
        ax.set_title("Molars per patient side", color=self.T["text"], fontsize=11)

        for a in axes:
            a.set_facecolor(self.T["mpl"])
            a.tick_params(colors=self.T["sub"], labelsize=8)
            for sp in a.spines.values():
                sp.set_color("#555")
            a.grid(axis="y", alpha=0.15)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.data_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _export_data(self):
        if not self.comp:
            messagebox.showinfo("Export", "Scan first.")
            return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                         initialfile="dataset_composition.xlsx")
        if not f:
            return
        SCAN.export_workbook(f, {
            "by_split": self.comp["by_split"], "by_side": self.comp["by_side"],
            "class_totals": self.comp["totals"],
            "provenance": pd.DataFrame([{"source": self.comp["source"]}]),
            "molar_level": self.comp["raw"],
        }, log=lambda m: self.set_status(m, "ok"))

    # ================================================================== Tab 2
    def _tab_metrics(self):
        t = self.tabs.tab("2 · Metrics")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(2, weight=3)
        t.grid_rowconfigure(4, weight=2)

        bar = ctk.CTkFrame(t, fg_color=self.T["card"])
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkButton(bar, text="Load metrics", width=130,
                      command=self._load_metrics).pack(side="left", padx=8, pady=8)
        self.inc_manu = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="fall back to manuscript values where no weights exist",
                        variable=self.inc_manu,
                        command=self._load_metrics).pack(side="left", padx=8)
        ctk.CTkButton(bar, text="Export to Excel", width=140,
                      command=self._export_metrics).pack(side="left", padx=8)
        self.metric_note = ctk.CTkLabel(bar, text="", text_color=self.T["warn"],
                                        font=ctk.CTkFont(size=11))
        self.metric_note.pack(side="left", padx=12)

        cols = ("model", "task", "task name", "source", "Precision", "Recall", "F1",
                "Dice", "mAP@0.5", "mAP@0.5:0.95", "Accuracy", "Cohen kappa",
                "Operating threshold", "P_A", "R_A", "F1_A", "n_A",
                "P_B", "R_B", "F1_B", "n_B", "P_C", "R_C", "F1_C", "n_C")
        f, self.metrics_tree = self._tree(t, cols, height=13)
        f.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self._all_trees.append(self.metrics_tree)

        cbar = ctk.CTkFrame(t, fg_color="transparent")
        cbar.grid(row=3, column=0, sticky="ew", padx=8)
        ctk.CTkLabel(cbar, text="Per-class 95% confidence intervals (Wilson score):",
                     text_color=self.T["sub"]).pack(side="left", padx=6)
        self.ci_model = ctk.CTkOptionMenu(cbar, values=["-"], width=170,
                                          command=lambda _=None: self._load_ci())
        self.ci_model.pack(side="left", padx=6)
        self.ci_task = ctk.CTkOptionMenu(
            cbar, values=[f"Task {i}" for i in SCAN.TASK_NAMES], width=110,
            command=lambda _=None: self._load_ci())
        self.ci_task.set("Task 7")
        self.ci_task.pack(side="left", padx=6)

        f2, self.ci_tree = self._tree(
            t, ("class", "support (n)", "precision", "precision 95% CI",
                "recall", "recall 95% CI", "F1", "source", "warning"), height=5)
        f2.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self._all_trees.append(self.ci_tree)

    def _load_metrics(self):
        df = SCAN.metrics_table(self.disc, include_manuscript=self.inc_manu.get())
        self.metrics_df = df
        self._fill(self.metrics_tree, df)
        n_m = int((df["source"] == "manuscript").sum()) if len(df) else 0
        n_meas = int((df["source"] == "measured").sum()) if len(df) else 0
        self.metric_note.configure(
            text=(f"{n_meas} measured · {n_m} from the manuscript (amber) — "
                  f"manuscript rows are the paper's own numbers, not results"
                  if n_m else f"{n_meas} measured rows, all from real evaluation runs"))
        models = sorted(df["model"].unique()) if len(df) else ["-"]
        self.ci_model.configure(values=models)
        if self.ci_model.get() not in models:
            self.ci_model.set(models[0])
        self._load_ci()
        self.set_status(f"metrics loaded: {len(df)} rows", "ok")

    def _load_ci(self):
        try:
            task = int(self.ci_task.get().split()[-1])
            model = self.ci_model.get()
            df = SCAN.per_class_ci_table(self.disc, model, task)
            self._fill(self.ci_tree, df)
        except Exception as e:  # noqa: BLE001
            self.set_status(f"CI table failed: {e}", "warn")

    def _export_metrics(self):
        if self.metrics_df.empty:
            messagebox.showinfo("Export", "Load the metrics first.")
            return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                         initialfile="metrics_all_models_tasks.xlsx")
        if not f:
            return
        sheets = {"ALL": self.metrics_df}
        if "source" in self.metrics_df:
            meas = self.metrics_df[self.metrics_df["source"] == "measured"]
            manu = self.metrics_df[self.metrics_df["source"] == "manuscript"]
            if len(meas):
                sheets["measured_only"] = meas
            if len(manu):
                sheets["manuscript_only_NOT_results"] = manu
        for m in sorted(self.metrics_df["model"].unique()):
            sheets[str(m)[:31]] = self.metrics_df[self.metrics_df["model"] == m]
        try:
            ci = SCAN.per_class_ci_table(self.disc, self.ci_model.get(),
                                         int(self.ci_task.get().split()[-1]))
            sheets["per_class_CI"] = ci
        except Exception:  # noqa: BLE001
            pass
        SCAN.export_workbook(f, sheets, log=lambda m: self.set_status(m, "ok"))

    # ================================================================== Tab 3
    def _tab_confusion(self):
        t = self.tabs.tab("3 · Confusion")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(t, fg_color=self.T["card"])
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkLabel(bar, text="Model").pack(side="left", padx=(10, 4), pady=10)
        self.cm_model = ctk.CTkOptionMenu(bar, values=["-"], width=180,
                                          command=lambda _=None: self._draw_confusion())
        self.cm_model.pack(side="left", padx=4)
        ctk.CTkLabel(bar, text="Task").pack(side="left", padx=(14, 4))
        self.cm_task = ctk.CTkOptionMenu(
            bar, values=[f"Task {i}" for i in SCAN.TASK_NAMES], width=110,
            command=lambda _=None: self._draw_confusion())
        self.cm_task.set("Task 7")
        self.cm_task.pack(side="left", padx=4)

        self.cm_show = ctk.StringVar(value="both")
        for label, val in [("numbers", "numbers"), ("percentage", "percent"),
                           ("numbers + percentage", "both")]:
            ctk.CTkRadioButton(bar, text=label, variable=self.cm_show, value=val,
                               command=self._draw_confusion).pack(side="left", padx=10)
        self.cm_norm = ctk.StringVar(value="row")
        ctk.CTkLabel(bar, text="% of").pack(side="left", padx=(14, 4))
        ctk.CTkOptionMenu(bar, values=["row", "column", "all"], width=100,
                          variable=self.cm_norm,
                          command=lambda _=None: self._draw_confusion()).pack(side="left")
        ctk.CTkButton(bar, text="Save PNG", width=100,
                      command=self._save_cm).pack(side="left", padx=12)
        ctk.CTkButton(bar, text="Export to Excel", width=140,
                      command=self._export_cm).pack(side="left", padx=4)

        self.cm_note = ctk.CTkLabel(t, text="", text_color=self.T["warn"],
                                    font=ctk.CTkFont(size=11), anchor="w",
                                    justify="left", wraplength=1500)
        self.cm_note.grid(row=1, column=0, sticky="ew", padx=14)

        self.cm_frame = ctk.CTkFrame(t, fg_color=self.T["card"])
        self.cm_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self._cm_fig = None

    def _draw_confusion(self):
        for w in self.cm_frame.winfo_children():
            w.destroy()
        model = self.cm_model.get()
        if model in ("-", ""):
            ctk.CTkLabel(self.cm_frame, text="Scan a project first.",
                         text_color=self.T["sub"]).pack(pady=40)
            return
        try:
            task = int(self.cm_task.get().split()[-1])
        except Exception:  # noqa: BLE001
            return
        info = SCAN.confusion(self.disc, model.replace(" (manuscript)", ""), task)
        m = info["matrix"]
        self.cm_note.configure(
            text=("MANUSCRIPT FALLBACK — " + info["note"]) if info["source"] == "manuscript"
            else info["note"],
            text_color=self.T["warn"] if info["source"] == "manuscript" else self.T["sub"])

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:  # noqa: BLE001
            ctk.CTkLabel(self.cm_frame, text=f"matplotlib unavailable: {e}").pack()
            return

        vals = m.to_numpy(dtype=float)
        known = ~np.isnan(vals)
        # When cells are unknown, the sum of the KNOWN cells is not the row total,
        # so normalising by it would print a triumphant 100% on a diagonal that is
        # only 100% because the errors are missing. Use the true support instead.
        row_tot = info.get("row_totals")
        if self.cm_norm.get() == "row":
            if row_tot:
                denom = np.array([[float(row_tot.get(str(ix).replace("true_", ""),
                                                     np.nan))]
                                  for ix in m.index])
            else:
                denom = np.nansum(vals, axis=1, keepdims=True)
        elif self.cm_norm.get() == "column":
            denom = np.nansum(vals, axis=0, keepdims=True)
        else:
            denom = np.array([[float(sum(row_tot.values()))]]) if row_tot \
                else np.array([[np.nansum(vals)]])
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(denom > 0, vals / denom * 100, np.nan)

        fig, ax = plt.subplots(figsize=(1.9 + 1.5 * vals.shape[1],
                                        1.9 + 1.2 * vals.shape[0]), dpi=100)
        fig.patch.set_facecolor(self.T["mpl"])
        ax.set_facecolor(self.T["mpl"])
        shown = np.where(known, pct, 0.0)
        im = ax.imshow(shown, cmap="viridis" if self.T["mode"] == "light" else "magma",
                       vmin=0, vmax=100)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if not known[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               hatch="////", fill=True,
                                               facecolor="#3a3a3a", edgecolor="#888",
                                               linewidth=1))
                    ax.text(j, i, "?", ha="center", va="center",
                            color="#ffd479", fontsize=15, fontweight="bold")
                    continue
                mode = self.cm_show.get()
                if mode == "numbers":
                    txt = f"{int(vals[i, j])}"
                elif mode == "percent":
                    txt = f"{pct[i, j]:.1f}%"
                else:
                    txt = f"{int(vals[i, j])}\n{pct[i, j]:.1f}%"
                # magma / viridis are dark at low values and light at high values,
                # so the readable text colour is the opposite of the usual guess
                ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                        color="#0d0d0d" if shown[i, j] > 55 else "#f2f2f2",
                        fontweight="bold")
        ax.set_xticks(range(vals.shape[1]))
        ax.set_xticklabels([str(c).replace("pred_", "") for c in m.columns],
                           color=self.T["text"])
        ax.set_yticks(range(vals.shape[0]))
        ax.set_yticklabels([str(i).replace("true_", "") for i in m.index],
                           color=self.T["text"])
        ax.set_xlabel("Predicted", color=self.T["text"])
        ax.set_ylabel("Reference", color=self.T["text"])
        title = f"{model} · Task {task} · {SCAN.TASK_DESC.get(task, '')}"
        if info["source"] == "manuscript":
            title += "   [manuscript fallback]"
        ax.set_title(title, color=self.T["text"], fontsize=11)
        ax.tick_params(colors=self.T["sub"])
        cb = fig.colorbar(im, ax=ax, fraction=0.045)
        cb.set_label(f"% of {self.cm_norm.get()}", color=self.T["text"])
        cb.ax.tick_params(colors=self.T["sub"])
        fig.tight_layout()
        self._cm_fig = fig
        self._cm_matrix = m
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _save_cm(self):
        if self._cm_fig is None:
            return
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile="confusion_matrix.png")
        if f:
            self._cm_fig.savefig(f, dpi=220, facecolor=self._cm_fig.get_facecolor())
            self.set_status(f"saved {f}", "ok")

    def _export_cm(self):
        f = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                         initialfile="confusion_matrices.xlsx")
        if not f:
            return
        sheets = {}
        pairs = SCAN.available_pairs(self.disc)
        for model, task, src in pairs:
            info = SCAN.confusion(self.disc, model.replace(" (manuscript)", ""), task)
            m = info["matrix"].copy()
            m.insert(0, "reference", m.index)
            m["source"] = info["source"]
            sheets[f"{model}_T{task}"[:31]] = m
        if not sheets:
            messagebox.showinfo("Export", "Nothing to export yet.")
            return
        SCAN.export_workbook(f, sheets, log=lambda m: self.set_status(m, "ok"))

    # ================================================================== Tab 4
    def _tab_test(self):
        t = self.tabs.tab("4 · Test")
        t.grid_columnconfigure(1, weight=1)
        t.grid_rowconfigure(0, weight=1)

        side = ctk.CTkScrollableFrame(t, width=330, fg_color=self.T["card"],
                                      label_text="Model and layers")
        side.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(side, text="Weights (.pt)", anchor="w").pack(fill="x", pady=(6, 2))
        self.pt_menu = ctk.CTkOptionMenu(side, values=["- scan first -"], width=290,
                                         command=self._select_pt)
        self.pt_menu.pack(fill="x")
        ctk.CTkButton(side, text="Browse for a .pt file",
                      command=self._browse_pt).pack(fill="x", pady=6)
        dv = ctk.CTkFrame(side, fg_color="transparent")
        dv.pack(fill="x")
        ctk.CTkLabel(dv, text="device", width=60, anchor="w").pack(side="left")
        self.dev_menu = ctk.CTkOptionMenu(dv, values=self._devices(), width=120,
                                          command=lambda _=None: self._select_pt(
                                              self.pt_menu.get()))
        self.dev_menu.pack(side="left", padx=4)
        self.pt_info = ctk.CTkLabel(side, text="", text_color=self.T["sub"],
                                    font=ctk.CTkFont(size=11), justify="left",
                                    anchor="w", wraplength=290)
        self.pt_info.pack(fill="x", pady=4)

        ctk.CTkLabel(side, text="Image", anchor="w").pack(fill="x", pady=(12, 2))
        ctk.CTkButton(side, text="Open an image",
                      command=self._open_image).pack(fill="x", pady=3)
        ctk.CTkButton(side, text="Pick from the dataset",
                      command=self._pick_from_dataset).pack(fill="x", pady=3)
        nav = ctk.CTkFrame(side, fg_color="transparent")
        nav.pack(fill="x", pady=3)
        ctk.CTkButton(nav, text="< prev", width=88,
                      command=lambda: self._step_image(-1)).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="next >", width=88,
                      command=lambda: self._step_image(1)).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="random", width=88,
                      command=lambda: self._step_image(0)).pack(side="left", padx=2)

        ctk.CTkLabel(side, text="Layers to display",
                     font=ctk.CTkFont(weight="bold"), anchor="w").pack(fill="x",
                                                                       pady=(14, 4))
        self.layer_vars = {}
        defaults = {"original", "heatmap_overlay", "contour_conf"}
        for key, label in VIZ.LAYER_LABELS:
            v = ctk.BooleanVar(value=key in defaults)
            self.layer_vars[key] = v
            ctk.CTkCheckBox(side, text=label, variable=v,
                            command=self._compose).pack(anchor="w", pady=2)
        rr = ctk.CTkFrame(side, fg_color="transparent")
        rr.pack(fill="x", pady=6)
        ctk.CTkButton(rr, text="all", width=88,
                      command=lambda: self._set_layers(True)).pack(side="left", padx=2)
        ctk.CTkButton(rr, text="none", width=88,
                      command=lambda: self._set_layers(False)).pack(side="left", padx=2)

        ctk.CTkLabel(side, text="Confidence threshold", anchor="w").pack(fill="x",
                                                                         pady=(12, 0))
        self.conf_lbl = ctk.CTkLabel(side, text="0.25", text_color=self.T["accent"])
        self.conf_lbl.pack()
        self.conf_slider = ctk.CTkSlider(side, from_=0.01, to=0.95, number_of_steps=94,
                                         command=self._conf_changed)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(fill="x")

        ctk.CTkLabel(side, text="Heat map opacity", anchor="w").pack(fill="x",
                                                                     pady=(10, 0))
        self.alpha_slider = ctk.CTkSlider(side, from_=0.05, to=0.9, number_of_steps=17,
                                          command=lambda _=None: self._run_inference())
        self.alpha_slider.set(0.45)
        self.alpha_slider.pack(fill="x")

        self.refine_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(side, text="refine mask inside the box (Otsu)",
                        variable=self.refine_var,
                        command=self._run_inference).pack(anchor="w", pady=8)
        ctk.CTkLabel(side, text="Columns", anchor="w").pack(fill="x")
        self.cols_menu = ctk.CTkOptionMenu(side, values=["1", "2", "3", "4"], width=90,
                                           command=lambda _=None: self._compose())
        self.cols_menu.set("3")
        self.cols_menu.pack(anchor="w", pady=4)

        ctk.CTkButton(side, text="RUN", height=38, fg_color=self.T["ok"],
                      command=self._run_inference).pack(fill="x", pady=(14, 4))
        ctk.CTkButton(side, text="Save the composite",
                      command=self._save_composite).pack(fill="x", pady=3)

        right = ctk.CTkFrame(t, fg_color=self.T["card"])
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.test_info = ctk.CTkLabel(right, text="Open an image and pick a .pt file.",
                                      anchor="w", justify="left",
                                      text_color=self.T["sub"],
                                      font=ctk.CTkFont(size=12), wraplength=1050)
        self.test_info.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        self.canvas_lbl = ctk.CTkLabel(right, text="")
        self.canvas_lbl.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._photo = None
        self._image_list = []
        self._image_idx = 0

    def _set_layers(self, on):
        for v in self.layer_vars.values():
            v.set(on)
        self._compose()

    def _conf_changed(self, v):
        self.conf_lbl.configure(text=f"{float(v):.2f}")
        if getattr(self, "_conf_job", None):
            self.after_cancel(self._conf_job)
        if self._alive:
            self._conf_job = self.after(320, self._run_inference)

    @staticmethod
    def _devices():
        opts = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                opts = [str(i) for i in range(torch.cuda.device_count())] + ["cpu"]
        except Exception:  # noqa: BLE001
            pass
        return opts

    def _browse_pt(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch weights", "*.pt")])
        if f:
            vals = list(self.pt_menu.cget("values"))
            if f not in vals:
                vals.append(f)
                self.pt_menu.configure(values=vals)
            self.pt_menu.set(f)
            self._select_pt(f)

    def _select_pt(self, path):
        if not path or path.startswith("-"):
            return
        dev = self.dev_menu.get() if hasattr(self, "dev_menu") else "cpu"
        self.predictor = VIZ.Predictor(path, device=dev)
        try:
            names = self.predictor.names
            self.pt_info.configure(
                text=f"task: {self.predictor.task}\nclasses: "
                     f"{', '.join(str(v) for v in names.values())}")
            self.set_status(f"loaded {Path(path).name}", "ok")
        except Exception as e:  # noqa: BLE001
            self.pt_info.configure(text=f"could not load: {e}")
            self.set_status(f"weight load failed: {e}", "bad")

    def _open_image(self):
        f = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if f:
            self._load_image(Path(f))

    def _pick_from_dataset(self):
        if not self.disc.images_dir:
            messagebox.showinfo("Dataset", "Scan a project folder first.")
            return
        self._image_list = sorted(p for p in self.disc.images_dir.iterdir()
                                  if p.suffix.lower() in SCAN.IMAGE_EXTS)
        if not self._image_list:
            messagebox.showinfo("Dataset", "No images found.")
            return
        self._image_idx = 0
        self._load_image(self._image_list[0])

    def _step_image(self, d):
        if not self._image_list:
            self._pick_from_dataset()
            return
        if d == 0:
            self._image_idx = int(np.random.randint(len(self._image_list)))
        else:
            self._image_idx = (self._image_idx + d) % len(self._image_list)
        self._load_image(self._image_list[self._image_idx])

    def _load_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            self.set_status(f"could not read {path}", "bad")
            return
        self.current_image = img
        self.current_path = path
        self.set_status(f"{path.name}  {img.shape[1]}x{img.shape[0]}", "ok")
        self._run_inference()

    def _run_inference(self):
        if self.current_image is None:
            return
        try:
            res = VIZ.render(self.current_image, self.predictor,
                             conf=float(self.conf_slider.get()),
                             alpha=float(self.alpha_slider.get()),
                             refine_mask=bool(self.refine_var.get()),
                             cmap=self.T["cmap"])
        except Exception as e:  # noqa: BLE001
            self.set_status(f"render failed: {e}", "bad")
            return
        self.current_render = res
        bits = [f"{Path(getattr(self, 'current_path', 'image')).name}",
                f"{len(res.detections)} detection(s)",
                f"heat map: {res.heatmap_method}"]
        if res.infer_ms:
            bits.append(f"{res.infer_ms:.0f} ms")
        for d in res.detections:
            side = "R" if (d.x1 + d.x2) / 2 < self.current_image.shape[1] / 2 else "L"
            bits.append(f"  · {d.label} conf {d.conf:.3f} → patient {side}")
        bits += res.notes
        self.test_info.configure(text="\n".join(bits))
        self._compose()

    def _compose(self):
        if not self.current_render:
            return
        keys = [k for k, _ in VIZ.LAYER_LABELS if self.layer_vars[k].get()]
        if not keys:
            self.canvas_lbl.configure(image=None, text="No layer selected.")
            return
        sheet = VIZ.grid(self.current_render.layers, keys,
                         cols=int(self.cols_menu.get()), cell_w=520)
        if sheet is None:
            return
        self._composite = sheet
        self.update_idletasks()
        maxw = max(420, self.canvas_lbl.winfo_width() - 20)
        maxh = max(320, self.canvas_lbl.winfo_height() - 20)
        h, w = sheet.shape[:2]
        s = min(maxw / w, maxh / h, 1.0)
        disp = cv2.resize(sheet, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA)
        try:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            self._photo = ctk.CTkImage(light_image=pil, dark_image=pil,
                                       size=(pil.width, pil.height))
            self.canvas_lbl.configure(image=self._photo, text="")
        except Exception as e:  # noqa: BLE001
            self.canvas_lbl.configure(text=f"pillow needed to display: {e}")

    def _save_composite(self):
        if getattr(self, "_composite", None) is None:
            return
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile="test_composite.png")
        if f:
            cv2.imwrite(f, self._composite)
            self.set_status(f"saved {f}", "ok")

    # ================================================================== Tab 5
    def _tab_proposed(self):
        t = self.tabs.tab("5 · Proposed")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(t, fg_color=self.T["card"])
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkButton(bar, text="Re-analyse", width=130,
                      command=self._build_proposed).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(bar, text="Save as text", width=130,
                      command=self._save_proposed).pack(side="left", padx=6)
        self.prop_box = ctk.CTkTextbox(t, font=ctk.CTkFont(family="Consolas", size=12),
                                       fg_color=self.T["card"],
                                       text_color=self.T["text"])
        self.prop_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _build_proposed(self):
        L = []
        add = L.append
        add("PROPOSED NEXT STEPS")
        add("=" * 78)
        add("")

        comp = self.comp
        if comp and len(comp.get("totals", [])):
            tot = comp["totals"]
            n = int(tot["molars"].sum())
            cC = int(tot[tot["class"] == "C"]["molars"].iloc[0]) if "C" in \
                tot["class"].values else 0
            add("1. CLASS C IS THE BINDING CONSTRAINT")
            add(f"   {cC} Class C molars out of {n} ({100*cC/max(1,n):.2f}%).")
            lo, hi = SCAN.wilson_ci(1, 1)
            add(f"   A random 10% test split leaves roughly {max(1, round(cC*0.1))} of")
            add("   them. With n = 1 the Wilson 95% CI for recall is "
                f"[{lo:.3f}, {hi:.3f}] — it spans")
            add("   almost the whole range, which is the honest way to say 'unknown'.")
            add("   Options, in order of scientific value:")
            add("     a. Collect more Class C cases. Nothing else actually fixes this.")
            add("     b. Report A and B per-class, and Class C as a count only.")
            add("     c. Merge B and C into a single 'non-routine' class and report a")
            add("        two-class problem honestly, if the clinical question allows it.")
            add("     d. k-fold cross-validation over the whole cohort so every Class C")
            add("        case is tested exactly once. This raises the effective test n")
            add("        from 1 to all 17 without inventing anything, at the cost of")
            add("        k times the training time. This is the strongest option that")
            add("        does not require new data.")
            add("   What does NOT fix it: augmentation. Copies of one molar are not")
            add("   independent cases and do not widen the evidence base.")
            add("")

        df = self.metrics_df
        if len(df):
            meas = df[df["source"] == "measured"] if "source" in df else df
            manu = df[df["source"] == "manuscript"] if "source" in df else df.iloc[0:0]
            add("2. WHAT IS ACTUALLY MEASURED RIGHT NOW")
            add(f"   {len(meas)} measured model/task rows, {len(manu)} still filled from")
            add("   the manuscript. Only the measured rows may go into the paper.")
            if len(meas) and "F1" in meas:
                best = meas.loc[meas["F1"].astype(float).idxmax()]
                add(f"   Best measured F1: {best['model']} on task {int(best['task'])} "
                    f"= {float(best['F1']):.4f}")
                for task in sorted(meas["task"].unique()):
                    sub = meas[meas["task"] == task].sort_values("F1", ascending=False)
                    if len(sub) >= 2:
                        gap = float(sub.iloc[0]["F1"]) - float(sub.iloc[1]["F1"])
                        if gap < 0.02:
                            add(f"   Task {int(task)}: {sub.iloc[0]['model']} leads "
                                f"{sub.iloc[1]['model']} by only {gap:.4f} — at this")
                            add("     test size that is not a real difference.")
            add("")

        add("3. OPERATING POINT")
        add("   Your task-1 curves peak at F1 0.79 at confidence 0.496, and precision")
        add("   at 0.25 is about 0.18. Report P/R/F1 at a threshold chosen on the")
        add("   VALIDATION split (conf_mode = auto), and state that threshold in the")
        add("   table caption. Quoting precision at 0.25 beside mAP 0.909 invites the")
        add("   obvious question.")
        add("")
        add("4. TRAINING LENGTH")
        add("   The 20-epoch run had validation mAP still climbing at the last epoch.")
        add("   Use 150-200 with patience 50 and let early stopping decide; otherwise")
        add("   every model comparison is a comparison of under-trained models.")
        add("")
        add("5. FIGURES FOR THE PAPER")
        add("   Essential, in the body: confusion matrix (Task 7), ROC with AUC and")
        add("   bootstrap CIs, one qualitative panel of detections.")
        add("   Appendix: per-task confusion matrices, training curves, the model")
        add("   registry table, and the Class C sensitivity analysis.")
        add("")
        add("6. WHAT A REVIEWER WILL CHECK FIRST")
        add("   - Do Table I and Table III come from the same test split? (they must)")
        add("   - Is the Class C support consistent everywhere? (it is 1, not 5)")
        add("   - Is the threshold for P/R/F1 stated?")
        add("   - Are the human and model Kappa values from the same sample? (they are")
        add("     not: 167-case subset vs 124-case test set — say so)")
        add("   - Does 'Dice' mean box Dice or mask Dice? (box; say so)")
        self.prop_box.delete("1.0", "end")
        self.prop_box.insert("1.0", "\n".join(L))

    def _save_proposed(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt",
                                         initialfile="proposed_next_steps.txt")
        if f:
            Path(f).write_text(self.prop_box.get("1.0", "end"), encoding="utf-8")
            self.set_status(f"saved {f}", "ok")

    # ================================================================== Tab 6
    def _tab_requirements(self):
        t = self.tabs.tab("6 · Requirements")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(t, fg_color=self.T["card"])
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkButton(bar, text="Check environment", width=160,
                      command=self._check_env).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(bar, text="Copy pip command", width=150,
                      command=self._copy_pip).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Write requirements.txt", width=170,
                      command=self._write_req).pack(side="left", padx=6)
        self.req_box = ctk.CTkTextbox(t, font=ctk.CTkFont(family="Consolas", size=12),
                                      fg_color=self.T["card"],
                                      text_color=self.T["text"])
        self.req_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    REQ = [("customtkinter", "5.2.0", "the GUI toolkit"),
           ("ultralytics", "8.3.0", "YOLO / RT-DETR training and inference"),
           ("torch", "2.0.0", "deep-learning backend"),
           ("torchvision", "0.15.0", "vision ops for torch"),
           ("cv2", "4.8.0", "opencv-python: image IO and drawing"),
           ("numpy", "1.24.0", "arrays"),
           ("pandas", "2.0.0", "tables and Excel"),
           ("matplotlib", "3.7.0", "charts"),
           ("sklearn", "1.2.0", "scikit-learn: metrics and Kappa"),
           ("scipy", "1.10.0", "statistics"),
           ("openpyxl", "3.1.0", "Excel writing"),
           ("PIL", "9.0.0", "pillow: image display in the GUI")]
    PIP = ("pip install customtkinter ultralytics torch torchvision opencv-python "
           "numpy pandas matplotlib scikit-learn scipy openpyxl pillow")

    def _check_env(self):
        import importlib
        L = [f"Python {sys.version.split()[0]}  ({sys.executable})", ""]
        missing = []
        for mod, want, why in self.REQ:
            try:
                m = importlib.import_module(mod)
                v = getattr(m, "__version__", "?")
                L.append(f"  OK      {mod:16s} {v:12s}  (need >= {want})   {why}")
            except Exception:  # noqa: BLE001
                missing.append(mod)
                L.append(f"  MISSING {mod:16s} {'':12s}  (need >= {want})   {why}")
        L.append("")
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                cap = f"{p.major}.{p.minor}"
                L.append(f"CUDA: {torch.cuda.get_device_name(0)}  "
                         f"{p.total_memory/1e9:.1f} GB  compute {cap}")
                L.append(f"torch {torch.__version__}  built for CUDA "
                         f"{getattr(torch.version, 'cuda', '?')}")
                if int(p.major) >= 10 and str(getattr(torch.version, "cuda", "")
                                              ).startswith("11"):
                    L.append("")
                    L.append("  ! MISMATCH: this GPU is Blackwell-class (compute >= 10)")
                    L.append("    but torch is built against CUDA 11. It will not run")
                    L.append("    on this card. Install the cu128 build:")
                    L.append("      pip install torch torchvision --index-url "
                             "https://download.pytorch.org/whl/cu128")
                if int(p.major) <= 5:
                    L.append("  note: Maxwell-class card — AMP saves memory but not time.")
            else:
                L.append("CUDA: not available — training would run on the CPU.")
        except Exception as e:  # noqa: BLE001
            L.append(f"CUDA check failed: {e}")
        L.append("")
        if missing:
            L.append("Install what is missing:")
            L.append("  pip install " + " ".join(
                {"cv2": "opencv-python", "sklearn": "scikit-learn",
                 "PIL": "pillow"}.get(m, m) for m in missing))
        else:
            L.append("Everything the dashboard needs is present.")
        L.append("")
        L.append("Full command:")
        L.append("  " + self.PIP)
        self.req_box.delete("1.0", "end")
        self.req_box.insert("1.0", "\n".join(L))
        self.set_status("environment checked",
                        "warn" if missing else "ok")

    def _copy_pip(self):
        self.clipboard_clear()
        self.clipboard_append(self.PIP)
        self.set_status("pip command copied to the clipboard", "ok")

    def _write_req(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt",
                                         initialfile="requirements.txt")
        if not f:
            return
        names = {"cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow"}
        Path(f).write_text("\n".join(
            f"{names.get(m, m)}>={v}" for m, v, _ in self.REQ) + "\n")
        self.set_status(f"saved {f}", "ok")

    # ================================================================= scanning
    def _pick_dir(self, var):
        d = filedialog.askdirectory(initialdir=var.get() or ".")
        if d:
            var.set(d)

    def rescan(self):
        self.set_status("scanning …")
        root = self.root_var.get()
        out = self.out_var.get()

        def work():
            try:
                disc = SCAN.scan(root, out)
                comp = SCAN.dataset_composition(disc)
                self.q.put(("scan", disc, comp))
            except Exception as e:  # noqa: BLE001
                self.q.put(("err", f"{e}\n{traceback.format_exc()}", None))

        threading.Thread(target=work, daemon=True).start()

    def _apply_scan(self, disc, comp):
        self.disc = disc
        self.comp = comp
        tot = comp.get("totals")
        if tot is not None and len(tot):
            counts = {r["class"]: int(r["molars"]) for _, r in tot.iterrows()}
            self.anim.count_up(self.stat_cards["images"],
                               tot.attrs.get("images_total", disc.n_images))
            self.anim.count_up(self.stat_cards["molars"], int(tot["molars"].sum()))
            for c in SCAN.CLASS_NAMES:
                self.anim.count_up(self.stat_cards[c], counts.get(c, 0))
            self.anim.count_up(self.stat_cards["both"],
                               tot.attrs.get("images_both_sides", 0))
        self.data_src.configure(text=f"source: {comp.get('source', '-')}")
        self._fill(self.data_tree, comp.get("by_split"))
        self._draw_data_charts()

        models = disc.models or ["yolo12s (manuscript)"]
        self.cm_model.configure(values=models)
        if self.cm_model.get() not in models:
            self.cm_model.set(models[0])
        self._draw_confusion()

        pts = []
        for model, per_task in disc.weights.items():
            for tid, p in sorted(per_task.items()):
                pts.append(str(p))
        pts += [str(p) for p in disc.loose_weights]
        self.pt_menu.configure(values=pts or ["- no .pt found -"])
        if pts:
            self.pt_menu.set(pts[0])

        self._load_metrics()
        self._build_proposed()
        self._check_env()
        n_w = sum(len(v) for v in disc.weights.values()) + len(disc.loose_weights)
        self.set_status(
            f"scan complete — {disc.n_images} images, {len(disc.models)} model folders, "
            f"{n_w} weight files", "ok")
        for line in disc.summary_lines():
            print(line)

    def _drain(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "scan":
                    self._apply_scan(a, b)
                elif kind == "err":
                    self.set_status("scan failed", "bad")
                    messagebox.showerror("Scan failed", str(a)[:1500])
        except queue.Empty:
            pass
        if self._alive:
            self._jobs.append(self.after(120, self._drain))


def open_dashboard(project_root: str = "", output_root: str = "", parent=None):
    """
    Launch the six analysis tabs.

    Called from opg_gui.py so training and analysis are one program: the pipeline
    GUI owns the Tk main loop and this opens as a second top-level window sharing
    it.  Paths are pre-filled from the caller and the scan starts immediately.
    """
    app = App()
    if project_root:
        app.root_var.set(project_root)
    if output_root:
        app.out_var.set(output_root)
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    if project_root:
        try:
            app.after(400, app.rescan)
        except Exception:  # noqa: BLE001
            pass
    return app


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()
