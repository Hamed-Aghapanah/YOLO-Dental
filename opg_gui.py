#!/usr/bin/env python
"""
opg_gui.py - CustomTkinter front-end, tabbed.

    Setup        project paths, models, tasks
    Hardware     GPU profile, suggested parameters, compatibility warnings
    Parameters   every hyper-parameter, manual mode
    Run          buttons, live log, status table with losses and metrics
    Results      all metrics per model x task, bar chart, best model per task

Every button calls opg_run.run() with the same arguments the command line would
build, so the GUI and opg_run.py produce identical output folders.
"""

from __future__ import annotations

import argparse
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
except ImportError:  # pragma: no cover
    raise SystemExit("customtkinter is missing.  pip install customtkinter")

import pandas as pd

import opg_hardware as HW
import opg_models as MODELS
import opg_run
from opg_tasks import ALL_TASKS, TASKS

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo12n", "yolo12s", "yolo12m",
               "yolo11n", "yolo11s", "yolo11m", "yolov10n", "yolov10s",
               "yolov9t", "yolov9s", "yolov9c", "yolov8n", "yolov8s",
               "yolov5nu", "yolov5su", "rtdetr-l", "rtdetr-x", "rtdetrv2-s"]
DEFAULT_MODELS = set(MODELS.DEFAULT_MODELS)

METRIC_CHOICES = ["F1", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95",
                  "Accuracy", "Dice", "Cohen kappa"]


class StopRequested(Exception):
    pass


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OPG Third-Molar Pipeline")
        self.geometry("1560x980")
        self.minsize(1220, 780)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.q: queue.Queue[str] = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker: threading.Thread | None = None
        self.gpu = HW.detect_gpu()
        self.results_df = pd.DataFrame()

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for name in ("Setup", "Hardware", "Parameters", "Run", "Results", "Analysis"):
            self.tabs.add(name)
        self._tab_setup()
        self._tab_parameters()
        self._tab_hardware()
        self._tab_run()
        self._tab_results()
        self._tab_analysis()
        self.tabs.set("Setup")
        self._alive = True
        self._drain_job = self.after(100, self._drain)

    # =============================================================== Setup tab
    def _tab_setup(self):
        t = self.tabs.tab("Setup")
        t.grid_columnconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)
        t.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(t)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        top.grid_columnconfigure(1, weight=1)

        def path_row(r, label, default, kind, hint=""):
            ctk.CTkLabel(top, text=label, anchor="w", width=210).grid(
                row=r, column=0, sticky="w", padx=6, pady=4)
            v = ctk.StringVar(value=default)
            ctk.CTkEntry(top, textvariable=v).grid(row=r, column=1, sticky="ew", padx=6)
            ctk.CTkButton(top, text="...", width=36,
                          command=lambda: self._pick(v, kind)).grid(row=r, column=2, padx=6)
            if hint:
                ctk.CTkLabel(top, text=hint, font=ctk.CTkFont(size=10),
                             text_color="#9ab", anchor="w").grid(
                    row=r + 1, column=1, sticky="w", padx=6)
            return v

        self.root_var = path_row(0, "Project folder", r"E:\project\0000_OPG\203-opg", "dir",
                                 "must contain images/, class.xlsx, localize.xlsx")
        self.out_var = path_row(2, "Output folder",
                                r"E:\project\0000_OPG\203-opg\output", "dir")
        self.cls_var = path_row(4, "class.xlsx (blank = auto)", "", "file")
        self.loc_var = path_row(6, "localize.xlsx (blank = auto)", "", "file")

        mf = ctk.CTkScrollableFrame(t, label_text="Models")
        mf.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.model_vars = {}
        fam_last = None
        for k in MODEL_ORDER:
            fam = MODELS.REGISTRY[k]["family"]
            if fam != fam_last:
                ctk.CTkLabel(mf, text=fam, font=ctk.CTkFont(weight="bold", size=12),
                             anchor="w").pack(fill="x", pady=(8, 2))
                fam_last = fam
            v = ctk.BooleanVar(value=k in DEFAULT_MODELS)
            self.model_vars[k] = v
            txt = k if MODELS.supports_classification(k) else \
                f"{k}   (no own classifier - tasks 3/4/5 substitute a YOLO cls head)"
            ctk.CTkCheckBox(mf, text=txt, variable=v,
                            font=ctk.CTkFont(size=11)).pack(anchor="w", pady=1)
        r = ctk.CTkFrame(mf, fg_color="transparent")
        r.pack(fill="x", pady=8)
        ctk.CTkButton(r, text="Paper Table I", width=110,
                      command=lambda: self._set_models(MODELS.PAPER_TABLE_I)).pack(
            side="left", padx=2)
        ctk.CTkButton(r, text="Default 3", width=90,
                      command=lambda: self._set_models(MODELS.DEFAULT_MODELS)).pack(
            side="left", padx=2)
        ctk.CTkButton(r, text="None", width=70,
                      command=lambda: self._set_models([])).pack(side="left", padx=2)

        tf = ctk.CTkScrollableFrame(t, label_text="Tasks")
        tf.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.task_vars = {}
        for k in ALL_TASKS:
            v = ctk.BooleanVar(value=True)
            self.task_vars[k] = v
            ctk.CTkCheckBox(tf, text=f"{k}. {TASKS[k]['desc']}", variable=v,
                            font=ctk.CTkFont(size=11)).pack(anchor="w", pady=2)
        r = ctk.CTkFrame(tf, fg_color="transparent")
        r.pack(fill="x", pady=8)
        ctk.CTkButton(r, text="All", width=60,
                      command=lambda: self._set_tasks(True)).pack(side="left", padx=2)
        ctk.CTkButton(r, text="None", width=60,
                      command=lambda: self._set_tasks(False)).pack(side="left", padx=2)
        ctk.CTkButton(r, text="Core 1,2,6,7", width=110,
                      command=lambda: self._set_tasks(False, {1, 2, 6, 7})).pack(
            side="left", padx=2)
        ctk.CTkLabel(tf, text="Tasks 8 and 9 are cascades: they train nothing new and\n"
                             "reuse the Task 1/4 and Task 2/5 weights.",
                     font=ctk.CTkFont(size=10), text_color="#9ab",
                     justify="left", anchor="w").pack(fill="x", pady=6)

    # ============================================================ Hardware tab
    def _tab_hardware(self):
        t = self.tabs.tab("Hardware")
        t.grid_columnconfigure(1, weight=1)
        t.grid_rowconfigure(3, weight=1)

        box = ctk.CTkFrame(t)
        box.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        g = self.gpu
        if g.get("available"):
            txt = (f"Detected: {g['name']}   {g['vram_gb']} GB   compute {g['capability']}\n"
                   f"torch {g['torch']}  /  CUDA {g['cuda']}")
        else:
            txt = ("No CUDA device visible to torch"
                   + (f"  ({g.get('error')})" if g.get("error") else "")
                   + "\nTraining would run on the CPU.")
        ctk.CTkLabel(box, text=txt, justify="left", anchor="w",
                     font=ctk.CTkFont(family="Consolas", size=12)).pack(
            fill="x", padx=10, pady=8)

        sel = ctk.CTkFrame(t)
        sel.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        ctk.CTkLabel(sel, text="Hardware profile", anchor="w").pack(side="left", padx=8)
        self.hw_var = ctk.StringVar(value=g.get("suggested_profile", "gtx980m"))
        ctk.CTkOptionMenu(sel, values=list(HW.PROFILES), variable=self.hw_var, width=180,
                          command=lambda _=None: self._refresh_hw()).pack(side="left", padx=6)
        self.hw_mode = ctk.StringVar(value="suggested")
        ctk.CTkRadioButton(sel, text="Use suggested", variable=self.hw_mode,
                           value="suggested", command=self._refresh_hw).pack(
            side="left", padx=12)
        ctk.CTkRadioButton(sel, text="Manual (Parameters tab wins)", variable=self.hw_mode,
                           value="manual", command=self._refresh_hw).pack(side="left", padx=6)
        self.slow_storage = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sel, text="slow / network storage", variable=self.slow_storage,
                        command=self._refresh_hw).pack(side="left", padx=12)
        ctk.CTkButton(sel, text="Apply to Parameters", command=self._apply_hw).pack(
            side="left", padx=10)

        ctk.CTkLabel(t, text="Suggestions are computed for the FIRST selected model, "
                             "because a YOLO-n and an RT-DETR do not fit the same batch.",
                     font=ctk.CTkFont(size=10), text_color="#9ab",
                     anchor="w").grid(row=2, column=0, columnspan=2, sticky="w", padx=14)

        self.hw_text = ctk.CTkTextbox(t, font=ctk.CTkFont(family="Consolas", size=12))
        self.hw_text.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        self.after(400, self._refresh_hw)

    def _first_model(self):
        sel = [k for k, v in self.model_vars.items() if v.get()]
        return sel[0] if sel else "yolo26n"

    def _refresh_hw(self):
        if not hasattr(self, "hw_text"):
            return
        try:
            s = HW.suggest(self.hw_var.get(), self._first_model(),
                           n_train_images=800, slow_storage=bool(self.slow_storage.get()))
        except Exception as e:  # noqa: BLE001
            self.hw_text.delete("1.0", "end")
            self.hw_text.insert("1.0", f"could not build a suggestion: {e}")
            return
        self._hw_suggestion = s
        p = s["profile"]
        lines = [p.label,
                 f"  architecture {p.arch}, compute {p.compute}, {p.vram_gb} GB VRAM",
                 "",
                 f"Suggested for '{self._first_model()}':"]
        lines += [f"    {k:14s} {v}" for k, v in s["overrides"].items()
                  if not k.startswith("_")]
        lines += ["", "Reasoning:"] + [f"  - {n}" for n in s["notes"]]
        warns = HW.compatibility_warnings(self.gpu) + s["warnings"]
        if warns:
            lines += ["", "WARNINGS:"] + [f"  ! {w}" for w in warns]
        if self.hw_mode.get() == "manual":
            lines += ["", "Manual mode: nothing is applied automatically. Press "
                          "'Apply to Parameters' to copy these values in."]
        self.hw_text.delete("1.0", "end")
        self.hw_text.insert("1.0", "\n".join(lines))

    def _apply_hw(self):
        s = getattr(self, "_hw_suggestion", None)
        if not s:
            return
        o = s["overrides"]
        for k in ("batch", "imgsz", "workers", "epochs", "patience", "save_period"):
            if k in o and k in self.p:
                self.p[k].set(str(o[k]))
        if "amp" in o:
            self.amp_var.set(bool(o["amp"]))
        self.cache_var.set(o.get("cache") or "off")
        messagebox.showinfo("Applied", "Suggested values copied into the Parameters tab.")
        self.tabs.set("Parameters")

    # =========================================================== Parameters tab
    def _tab_parameters(self):
        t = self.tabs.tab("Parameters")
        t.grid_columnconfigure(0, weight=1)
        t.grid_columnconfigure(1, weight=1)
        t.grid_rowconfigure(0, weight=1)
        left = ctk.CTkScrollableFrame(t, label_text="Training")
        left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        right = ctk.CTkScrollableFrame(t, label_text="Data, evaluation and search")
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.p = {}

        def num(parent, key, label, default, hint=""):
            ctk.CTkLabel(parent, text=label, anchor="w",
                         font=ctk.CTkFont(size=12)).pack(fill="x", pady=(8, 0))
            v = ctk.StringVar(value=str(default))
            self.p[key] = v
            ctk.CTkEntry(parent, textvariable=v).pack(fill="x")
            if hint:
                ctk.CTkLabel(parent, text=hint, font=ctk.CTkFont(size=10),
                             text_color="#9ab", anchor="w", justify="left").pack(fill="x")

        num(left, "epochs", "Epochs", 200)
        num(left, "batch", "Batch size", 8)
        num(left, "imgsz", "Image size (detection)", 640)
        num(left, "imgsz_cls", "Image size (classification)", 224)
        num(left, "patience", "Early-stopping patience", 50)
        num(left, "save_period", "Checkpoint every N epochs", 10,
            "last.pt is what resume uses. Smaller = less lost on a crash.")
        num(left, "lr0", "Initial LR (lr0)", 0.01)
        num(left, "lrf", "Final LR factor (lrf)", 0.01)
        num(left, "device", "Device", "0", "'0' = first GPU, 'cpu', or '0,1'")
        num(left, "workers", "Dataloader workers", 8)
        num(left, "seed", "Random seed", 42)

        ctk.CTkLabel(left, text="Optimizer", anchor="w").pack(fill="x", pady=(8, 0))
        self.opt_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(left, values=["auto", "SGD", "Adam", "AdamW"],
                          variable=self.opt_var).pack(fill="x")
        ctk.CTkLabel(left, text="'auto' lets Ultralytics choose; your log shows it then\n"
                                "ignores lr0 and momentum and picks AdamW lr=0.002.\n"
                                "Set SGD or AdamW explicitly if lr0 must be honoured.",
                     font=ctk.CTkFont(size=10), text_color="#9ab",
                     justify="left", anchor="w").pack(fill="x")

        self.amp_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="AMP (mixed precision)",
                        variable=self.amp_var).pack(anchor="w", pady=(10, 2))
        self.resume_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Resume from last.pt if a run was interrupted",
                        variable=self.resume_var).pack(anchor="w", pady=2)
        ctk.CTkLabel(left, text="Dataset cache", anchor="w").pack(fill="x", pady=(8, 0))
        self.cache_var = ctk.StringVar(value="off")
        ctk.CTkOptionMenu(left, values=["off", "ram", "disk"],
                          variable=self.cache_var).pack(fill="x")
        ctk.CTkLabel(left, text="'disk' decodes each image once and reuses it; this is the\n"
                                "single biggest win when the data sits on slow storage.",
                     font=ctk.CTkFont(size=10), text_color="#9ab",
                     justify="left", anchor="w").pack(fill="x")

        ctk.CTkLabel(right, text="Data split", font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(fill="x")
        num(right, "train_frac", "Train fraction", 0.8)
        num(right, "val_frac", "Validation fraction", 0.1)
        num(right, "min_c_test", "Force N Class-C images into test", 0,
            "0 = purely random stratified split (what the manuscript reports).\n"
            "Any value > 0 is a NON-RANDOM allocation and must be declared.")
        num(right, "min_c_val", "Force N Class-C images into val", 0)

        ctk.CTkLabel(right, text="Class imbalance", font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(fill="x", pady=(14, 0))
        self.focal_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(right, text="Focal loss on the detection head",
                        variable=self.focal_var).pack(anchor="w", pady=2)
        ctk.CTkLabel(right, text="Not applied to RT-DETR, which uses its own loss.",
                     font=ctk.CTkFont(size=10), text_color="#9ab", anchor="w").pack(fill="x")
        num(right, "focal_alpha", "Focal alpha", 0.75)
        num(right, "focal_gamma", "Focal gamma", 2.0)
        num(right, "class_c_aug", "Augmented copies per Class-C image", 8)

        ctk.CTkLabel(right, text="Evaluation", font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(right, text="Operating point for P/R/F1", anchor="w").pack(
            fill="x", pady=(6, 0))
        self.confmode_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(right, values=["auto", "fixed"],
                          variable=self.confmode_var).pack(fill="x")
        ctk.CTkLabel(right,
                     text="auto: the F1-optimal threshold is derived on the VALIDATION\n"
                          "split and applied to test. Your task-1 curves peak at F1 0.79\n"
                          "at confidence 0.496, and precision at 0.25 is only about 0.18,\n"
                          "so a fixed 0.25 prints precision ~0.2 beside mAP 0.909.",
                     font=ctk.CTkFont(size=10), text_color="#9ab",
                     justify="left", anchor="w").pack(fill="x")
        num(right, "conf_eval", "Fixed threshold (used when mode = fixed)", 0.25)
        num(right, "roi_pad", "ROI crop padding", 0.25)
        num(right, "fliplr", "Horizontal flip probability", 0.0,
            "Keep 0: mirroring an OPG swaps patient L and R.")
        num(right, "bootstrap", "Bootstrap iterations", 1000)
        self.bench_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(right, text="Benchmark inference speed for the table",
                        variable=self.bench_var).pack(anchor="w", pady=6)

        ctk.CTkLabel(right, text="Hyper-parameter search", font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(fill="x", pady=(14, 0))
        num(right, "tune_tasks", "Tasks to tune", "1,7")
        num(right, "tune_trials", "Random-search trials", 20)
        num(right, "tune_epochs", "Epochs per trial", 40)

    # ================================================================= Run tab
    def _tab_run(self):
        t = self.tabs.tab("Run")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(3, weight=3)
        t.grid_rowconfigure(5, weight=2)

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        for txt, st in [("Status", ["status"]), ("1. Prepare", ["prepare"]),
                        ("2. Tune", ["tune"]), ("3. Train", ["train"]),
                        ("4. Test", ["test"]), ("5. Summary", ["summary"])]:
            ctk.CTkButton(bar, text=txt, width=104,
                          command=lambda s=st: self._go(s)).pack(side="left", padx=3)

        bar2 = ctk.CTkFrame(t, fg_color="transparent")
        bar2.grid(row=1, column=0, sticky="ew", padx=6, pady=2)
        ctk.CTkButton(bar2, text="RUN EVERYTHING", fg_color="#1f7a3d",
                      hover_color="#166030",
                      command=lambda: self._go(["prepare", "train", "test", "summary"])
                      ).pack(side="left", padx=3)
        ctk.CTkButton(bar2, text="+ TUNE", fg_color="#1f5f7a", hover_color="#164a60",
                      command=lambda: self._go(
                          ["prepare", "tune", "train", "test", "summary"])
                      ).pack(side="left", padx=3)
        ctk.CTkButton(bar2, text="Stop", fg_color="#8a2f2f", hover_color="#6d2525",
                      command=self._stop).pack(side="left", padx=3)
        self.force_train = ctk.BooleanVar(value=False)
        self.force_test = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar2, text="force retrain",
                        variable=self.force_train).pack(side="left", padx=10)
        ctk.CTkCheckBox(bar2, text="force re-test",
                        variable=self.force_test).pack(side="left", padx=6)

        bar3 = ctk.CTkFrame(t, fg_color="transparent")
        bar3.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
        ctk.CTkLabel(bar3, text="Open:").pack(side="left", padx=(6, 4))
        for label, sub in [("output root", ""), ("results", "results"),
                           ("datasets", "datasets"),
                           ("reviewer package", "reviewer_package")]:
            ctk.CTkButton(bar3, text=label, width=126,
                          command=lambda s=sub: self._open(s)).pack(side="left", padx=3)
        ctk.CTkButton(bar3, text="open a model folder", width=150,
                      command=self._open_model).pack(side="left", padx=8)

        self.log_box = ctk.CTkTextbox(t, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=3, column=0, sticky="nsew", padx=6, pady=4)

        sb = ctk.CTkFrame(t, fg_color="transparent")
        sb.grid(row=4, column=0, sticky="ew", padx=6)
        ctk.CTkLabel(sb, text="Status  (losses and metrics of the last completed epoch)",
                     font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkButton(sb, text="Refresh", width=80,
                      command=self._refresh_status).pack(side="left", padx=8)
        self.auto_refresh = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sb, text="auto every 20 s",
                        variable=self.auto_refresh).pack(side="left", padx=6)

        cols = ("model", "task", "name", "trained", "evaluated", "epochs",
                "box_loss", "cls_loss", "dfl_loss", "val_P", "val_R",
                "val_mAP50", "val_mAP50_95", "val_acc", "test_F1", "test_mAP50",
                "minutes")
        self.status_tree = self._make_tree(t, cols, row=5)
        self.after(20000, self._auto_status)

        self.progress = ctk.CTkProgressBar(t)
        self.progress.grid(row=6, column=0, sticky="ew", padx=6, pady=(4, 8))
        self.progress.set(0)

        self._log("Ready.\n"
                  "Order: Prepare -> (optional) Tune -> Train -> Test -> Summary.\n"
                  "Finished training is skipped; an interrupted run resumes from last.pt.\n"
                  "Results land in <output>/results/<model>/<task>/.\n")

    def _make_tree(self, parent, cols, row):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, sticky="nsew", padx=6, pady=4)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:  # noqa: BLE001
            pass
        style.configure("Treeview", background="#242424", foreground="#e6e6e6",
                        fieldbackground="#242424", rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", background="#1a1a1a",
                        foreground="#cfe3ff", relief="flat")
        style.map("Treeview", background=[("selected", "#2a5d9f")])
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(62, min(190, 9 * len(str(c)) + 46)), anchor="center")
        vs = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        return tree

    def _auto_status(self):
        if self.auto_refresh.get() and self.worker and self.worker.is_alive():
            self._refresh_status(quiet=True)
        self._status_job = self.after(20000, self._auto_status)

    def _refresh_status(self, quiet=False):
        try:
            import opg_tasks as T
            out = Path(self.out_var.get())
            pipe = T.Pipeline(out / "datasets", out / "results",
                              T.TrainConfig(epochs=int(self.p["epochs"].get())),
                              log=lambda *_a, **_k: None)
            models = [k for k, v in self.model_vars.items() if v.get()]
            tasks = [k for k, v in self.task_vars.items() if v.get()]
            df = pipe.status_table(models, tasks)
            for i in self.status_tree.get_children():
                self.status_tree.delete(i)
            for _, r in df.iterrows():
                self.status_tree.insert(
                    "", "end", values=[r.get(c, "") for c in self.status_tree["columns"]])
        except Exception as e:  # noqa: BLE001
            if not quiet:
                self._log(f"status refresh failed: {e}")

    # ============================================================= Results tab
    def _tab_results(self):
        t = self.tabs.tab("Results")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(1, weight=3)
        t.grid_rowconfigure(3, weight=4)

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ctk.CTkButton(bar, text="Load / refresh results", width=170,
                      command=self._load_results).pack(side="left", padx=4)
        ctk.CTkLabel(bar, text="Metric:").pack(side="left", padx=(16, 4))
        self.metric_var = ctk.StringVar(value="F1")
        ctk.CTkOptionMenu(bar, values=METRIC_CHOICES, variable=self.metric_var, width=150,
                          command=lambda _=None: self._draw_chart()).pack(side="left")
        ctk.CTkLabel(bar, text="Task:").pack(side="left", padx=(16, 4))
        self.chart_task = ctk.StringVar(value="all tasks")
        self.chart_task_menu = ctk.CTkOptionMenu(
            bar, values=["all tasks"], variable=self.chart_task, width=250,
            command=lambda _=None: self._draw_chart())
        self.chart_task_menu.pack(side="left")
        ctk.CTkButton(bar, text="Export table to CSV", width=160,
                      command=self._export_results).pack(side="left", padx=16)

        cols = ("Model", "Task", "Task name", "Precision", "Recall", "F1", "Dice",
                "mAP@0.5", "mAP@0.5:0.95", "Accuracy", "Cohen kappa",
                "Operating threshold", "Precision_C", "Recall_C", "Support_C",
                "Parameters (M)", "GFLOPs", "Epochs run", "Train minutes")
        self.res_tree = self._make_tree(t, cols, row=1)

        self.best_label = ctk.CTkLabel(
            t, text="Best model per task appears here once results are loaded.",
            anchor="w", justify="left", font=ctk.CTkFont(family="Consolas", size=12))
        self.best_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 2))

        self.chart_frame = ctk.CTkFrame(t)
        self.chart_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=6)

    def _results_path(self):
        return Path(self.out_var.get()) / "tasks_comparison_with_metrics.csv"

    def _load_results(self):
        f = self._results_path()
        if f.exists():
            self.results_df = pd.read_csv(f)
        else:
            xl = f.with_suffix(".xlsx")
            if not xl.exists():
                messagebox.showinfo(
                    "Results", f"No summary found at\n{f}\n\nRun stage 5 (Summary) first.")
                return
            try:
                self.results_df = pd.read_excel(xl, sheet_name="ALL_MODELS")
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Results", str(e))
                return

        df = self.results_df
        for i in self.res_tree.get_children():
            self.res_tree.delete(i)
        for _, r in df.iterrows():
            self.res_tree.insert("", "end",
                                 values=[r.get(c, "") for c in self.res_tree["columns"]])

        names = ["all tasks"]
        if "Task" in df.columns:
            for tid in sorted(df["Task"].dropna().unique()):
                nm = (df[df["Task"] == tid]["Task name"].iloc[0]
                      if "Task name" in df.columns else "")
                names.append(f"{int(tid)} - {nm}")
        self.chart_task_menu.configure(values=names)
        if self.chart_task.get() not in names:
            self.chart_task.set("all tasks")
        self._draw_chart()

    @staticmethod
    def _numeric(s):
        return pd.to_numeric(s, errors="coerce")

    def _update_best(self):
        df, metric = self.results_df, self.metric_var.get()
        if df.empty or metric not in df.columns:
            return
        d = df.copy()
        d["_v"] = self._numeric(d[metric])
        d = d.dropna(subset=["_v"])
        if d.empty:
            self.best_label.configure(
                text=f"No numeric values of {metric} - that metric is 'n.a.' for "
                     f"these tasks (mAP is undefined for classification, "
                     f"accuracy for detection).")
            return
        lines = [f"Best model per task by {metric}:"]
        for tid in sorted(d["Task"].unique()):
            sub = d[d["Task"] == tid].sort_values("_v", ascending=False)
            top = sub.iloc[0]
            name = str(top.get("Task name", ""))
            line = (f"  Task {int(tid)} {name:22s}  {str(top['Model']):10s} "
                    f"{metric} = {top['_v']:.4f}")
            if len(sub) > 1:
                second = sub.iloc[1]
                gap = top["_v"] - second["_v"]
                line += (f"   next: {second['Model']} {second['_v']:.4f}"
                         f"  gap {gap:.4f}")
                if gap < 0.02:
                    line += "  <- within noise at this test size"
            lines.append(line)
        lines.append("")
        lines.append("A gap of a few thousandths on a ~124-image test set is not a real "
                     "difference. Check the Significance sheet before naming a winner.")
        self.best_label.configure(text="\n".join(lines))

    def _draw_chart(self):
        df, metric = self.results_df, self.metric_var.get()
        for w in self.chart_frame.winfo_children():
            w.destroy()
        if df.empty or metric not in df.columns:
            ctk.CTkLabel(self.chart_frame,
                         text="Load results first (stage 5 writes the summary).").pack(
                pady=20)
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:  # noqa: BLE001
            ctk.CTkLabel(self.chart_frame, text=f"matplotlib unavailable: {e}").pack()
            return

        d = df.copy()
        d["_v"] = self._numeric(d[metric])
        d = d.dropna(subset=["_v"])
        sel = self.chart_task.get()
        if sel != "all tasks":
            d = d[d["Task"] == int(sel.split(" - ")[0])]
        if d.empty:
            ctk.CTkLabel(self.chart_frame,
                         text=f"No values of {metric} for this selection.").pack(pady=20)
            self._update_best()
            return

        models = sorted(d["Model"].unique())
        tasks = sorted(d["Task"].unique())
        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=100)
        fig.patch.set_facecolor("#242424")
        ax.set_facecolor("#1c1c1c")
        width = 0.8 / max(1, len(models))
        x = np.arange(len(tasks))
        palette = ["#4c9be8", "#59c17a", "#e8a24c", "#c86bd8", "#e05f5f",
                   "#7fd4d4", "#b0b0b0"]
        for i, m in enumerate(models):
            vals = []
            for tid in tasks:
                sub = d[(d["Model"] == m) & (d["Task"] == tid)]["_v"]
                vals.append(float(sub.iloc[0]) if len(sub) else np.nan)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
                   label=str(m), color=palette[i % len(palette)])
        for j, tid in enumerate(tasks):
            sub = d[d["Task"] == tid]
            if sub.empty:
                continue
            best = sub.loc[sub["_v"].idxmax()]
            ax.text(x[j], float(best["_v"]) + 0.015, f"best: {best['Model']}",
                    ha="center", va="bottom", fontsize=7, color="#ffd479")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{int(t)}" for t in tasks], color="#e6e6e6")
        ax.set_ylabel(metric, color="#e6e6e6")
        ax.set_title(f"{metric} by task and model", color="#e6e6e6")
        ax.tick_params(colors="#c9c9c9")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(1.05, float(d["_v"].max()) * 1.18))
        ax.legend(fontsize=8, ncol=min(len(models), 7), facecolor="#1c1c1c",
                  edgecolor="#444", labelcolor="#e6e6e6")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._update_best()

    def _export_results(self):
        if self.results_df.empty:
            messagebox.showinfo("Export", "Load the results first.")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv",
                                         initialfile="results_table.csv")
        if f:
            self.results_df.to_csv(f, index=False)
            self._log(f"results table exported -> {f}")

    # ============================================================== Analysis tab
    def _tab_analysis(self):
        """Data / Metrics / Confusion / Test / Proposed / Requirements, from
        opg_dashboard, launched against the paths already set on the Setup tab."""
        t = self.tabs.tab("Analysis")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 4))
        ctk.CTkButton(bar, text="Open the analysis dashboard", width=240,
                      fg_color="#1f5f7a", hover_color="#164a60",
                      command=self._open_dashboard).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Export EVERYTHING to Excel", width=230,
                      fg_color="#1f7a3d", hover_color="#166030",
                      command=self._export_all).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Open the EXPORT folder", width=180,
                      command=lambda: self._open("EXPORT")).pack(side="left", padx=4)

        ctk.CTkLabel(t, text="The dashboard opens as a second window sharing this "
                             "program's event loop; the project and output paths from "
                             "the Setup tab are filled in and the scan starts on its own.",
                     font=ctk.CTkFont(size=11), text_color="#9ab",
                     anchor="w", justify="left").grid(row=1, column=0, sticky="ew",
                                                      padx=14)

        self.an_text = ctk.CTkTextbox(t, font=ctk.CTkFont(family="Consolas", size=12))
        self.an_text.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self.an_text.insert("1.0",
            "Analysis tabs\n"
            "  1 Data          class A/B/C across train / val / test, per patient side\n"
            "  2 Metrics       every model x task, with per-class Wilson 95% CIs\n"
            "  3 Confusion     one matrix per model and task\n"
            "  4 Test          run any .pt on any image with selectable overlays\n"
            "  5 Proposed      what the numbers say and what to run next\n"
            "  6 Requirements  environment check with copyable fix commands\n\n"
            "'Export EVERYTHING to Excel' runs opg_export_all.py: it evaluates every\n"
            "(model, task) that has weights but has not been tested, then writes\n"
            "  <output>/EXPORT/tasks_comparison_with_metrics.xlsx\n"
            "  <output>/EXPORT/confusion_matrices.xlsx  + one PNG per pair\n"
            "  <output>/EXPORT/per_class_metrics.xlsx   (Wilson 95% CIs)\n"
            "  <output>/EXPORT/PREDICTIONS_FOR_KAPPA.csv\n\n"
            "Confusion matrices cannot come from the weights alone: the model has to be\n"
            "run over the held-out split and the predictions scored. If your Metrics tab\n"
            "says '0 measured', that step has not happened yet - press Export and it will.\n")

    def _open_dashboard(self):
        try:
            import opg_dashboard as DASH
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Dashboard unavailable", str(e))
            return
        try:
            self._dash = DASH.open_dashboard(self.root_var.get(), self.out_var.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not open the dashboard", traceback.format_exc())

    def _export_all(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "A job is already running.")
            return
        import argparse as _ap
        args = _ap.Namespace(root=self.root_var.get(), out=self.out_var.get(),
                             device=self.p["device"].get(),
                             conf_mode=self.confmode_var.get(),
                             conf_eval=float(self.p["conf_eval"].get()),
                             bootstrap=int(self.p["bootstrap"].get()),
                             force=bool(self.force_test.get()),
                             models="", tasks="")
        self.tabs.set("Run")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self._log("\n" + "=" * 70)
        self._log(">>> exporting every metric and confusion matrix")
        self._log("=" * 70)

        def log(m=""):
            if self.stop_flag.is_set():
                raise StopRequested()
            self.q.put(str(m))

        def work():
            try:
                import opg_export_all as EX
                EX.run(args, log=log)
                self.q.put("\n>>> export finished")
            except StopRequested:
                self.q.put("\n>>> stopped by user")
            except Exception as e:  # noqa: BLE001
                self.q.put(f"\n!!! ERROR: {e}\n{traceback.format_exc()}")
            finally:
                self.after(0, self._finish)

        self.stop_flag.clear()
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # =============================================================== utilities
    def _pick(self, var, kind):
        p = (filedialog.askopenfilename(
            filetypes=[("Spreadsheet", "*.xlsx *.xls *.csv"), ("All", "*.*")])
            if kind == "file" else filedialog.askdirectory(initialdir=var.get() or "."))
        if p:
            var.set(p)

    def _set_models(self, keys):
        keys = set(keys)
        for k, v in self.model_vars.items():
            v.set(k in keys)
        self._refresh_hw()

    def _set_tasks(self, value, only=None):
        for t, v in self.task_vars.items():
            v.set(value if only is None else (t in only))

    def _log(self, m=""):
        self.q.put(str(m))

    def _drain(self):
        try:
            while True:
                self.log_box.insert("end", self.q.get_nowait() + "\n")
                self.log_box.see("end")
        except queue.Empty:
            pass
        if getattr(self, "_alive", True):
            self._drain_job = self.after(120, self._drain)

    def _open(self, sub=""):
        import subprocess
        import sys
        p = Path(self.out_var.get())
        if sub:
            p = p / sub
        if not p.exists():
            messagebox.showinfo(
                "Not there yet",
                f"{p}\n\ndoes not exist yet.\n\n"
                "Model folders appear under <output>/results/<model>/<task>/ only after "
                "training has started. Run Prepare, then Train.")
            return
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _open_model(self):
        res = Path(self.out_var.get()) / "results"
        if not res.exists():
            messagebox.showinfo("Not there yet",
                                f"{res}\n\ndoes not exist yet. Train something first.")
            return
        subs = sorted(d.name for d in res.iterdir() if d.is_dir())
        if not subs:
            messagebox.showinfo("Empty", f"{res} has no model folders yet.")
            return
        win = ctk.CTkToplevel(self)
        win.title("Open a model folder")
        win.geometry("440x170")
        ctk.CTkLabel(win, text=f"Found in {res}:").pack(pady=8)
        var = ctk.StringVar(value=subs[0])
        ctk.CTkOptionMenu(win, values=subs, variable=var, width=320).pack(pady=6)
        ctk.CTkButton(win, text="Open",
                      command=lambda: (self._open(f"results/{var.get()}"),
                                       win.destroy())).pack(pady=10)

    def _stop(self):
        self.stop_flag.set()
        self._log("\n>>> stop requested - halting at the next step boundary\n")

    # ===================================================================== run
    def _args(self, stages) -> argparse.Namespace:
        g = lambda k, cast=float: cast(self.p[k].get())
        models = [m for m, v in self.model_vars.items() if v.get()]
        tasks = [t for t, v in self.task_vars.items() if v.get()]
        if not models:
            raise ValueError("select at least one model")
        if not tasks and stages != ["prepare"]:
            raise ValueError("select at least one task")
        return argparse.Namespace(
            root=self.root_var.get(), out=self.out_var.get(),
            class_xlsx=self.cls_var.get(), localize_xlsx=self.loc_var.get(),
            models=",".join(models), tasks=",".join(map(str, tasks)) or "1",
            stages=",".join(stages), conflict_policy="hardest",
            roi_pad=g("roi_pad"), class_c_aug=g("class_c_aug", int),
            min_c_test=g("min_c_test", int), min_c_val=g("min_c_val", int),
            train_frac=g("train_frac"), val_frac=g("val_frac"),
            epochs=g("epochs", int), batch=g("batch", int), imgsz=g("imgsz", int),
            imgsz_cls=g("imgsz_cls", int), patience=g("patience", int),
            optimizer=self.opt_var.get(), lr0=g("lr0"), lrf=g("lrf"),
            device=self.p["device"].get(), workers=g("workers", int),
            seed=g("seed", int), fliplr=g("fliplr"),
            no_focal=not self.focal_var.get(),
            focal_alpha=g("focal_alpha"), focal_gamma=g("focal_gamma"),
            conf_eval=g("conf_eval"), conf_mode=self.confmode_var.get(),
            no_resume=not self.resume_var.get(),
            amp="on" if self.amp_var.get() else "off",
            cache="" if self.cache_var.get() == "off" else self.cache_var.get(),
            save_period=g("save_period", int),
            benchmark_speed=bool(self.bench_var.get()),
            hardware=self.hw_var.get() if self.hw_mode.get() == "suggested" else "",
            auto_hardware=False, slow_storage=bool(self.slow_storage.get()),
            list_models=False,
            tune_tasks=self.p["tune_tasks"].get(), tune_trials=g("tune_trials", int),
            tune_epochs=g("tune_epochs", int), tune_model="",
            bootstrap=g("bootstrap", int),
            force_train=bool(self.force_train.get()),
            force_test=bool(self.force_test.get()))

    def _go(self, stages):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "A job is already running.")
            return
        try:
            args = self._args(stages)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid settings", str(e))
            return
        if "prepare" in stages and not (Path(args.root) / "images").exists():
            messagebox.showerror("Missing", f"{args.root}\n\nhas no images/ folder.")
            return

        self.tabs.set("Run")
        self.stop_flag.clear()
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self._log("\n" + "=" * 70)
        self._log(f">>> {', '.join(stages)}  models={args.models}  tasks={args.tasks}")
        self._log("=" * 70)

        def log(m=""):
            if self.stop_flag.is_set():
                raise StopRequested()
            self.q.put(str(m))

        def work():
            try:
                opg_run.run(args, log=log)
                self.q.put("\n>>> finished")
            except StopRequested:
                self.q.put("\n>>> stopped by user")
            except Exception as e:  # noqa: BLE001
                self.q.put(f"\n!!! ERROR: {e}\n{traceback.format_exc()}")
            finally:
                self.after(0, self._finish)

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _finish(self):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(1.0)
        self._refresh_status(quiet=True)
        try:
            if self._results_path().exists() or \
                    self._results_path().with_suffix(".xlsx").exists():
                self._load_results()
        except Exception:  # noqa: BLE001
            pass


    def _on_close(self):
        """Cancel the repeating status/log timers before the widgets go away."""
        self._alive = False
        self.stop_flag.set()
        for attr in ("_drain_job", "_status_job"):
            j = getattr(self, attr, None)
            if j:
                try:
                    self.after_cancel(j)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.quit()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()
