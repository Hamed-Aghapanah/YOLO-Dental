#!/usr/bin/env python
"""
opg_export_all.py
=================
ONE COMMAND that produces request 1: an Excel workbook and a confusion matrix for
every model and every task.

    python opg_export_all.py --root E:\\project\\0000_OPG\\203-opg \\
                             --out  E:\\project\\0000_OPG\\203-opg\\output

WHY THIS SCRIPT EXISTS
----------------------
Your dashboard scan reported:

    models found : yolo11s, yolo12s, yolo26n, yolo26s
    weight files : 8 inside results/
    Metrics tab  : 0 measured, 13 from the manuscript

Eight checkpoints exist, but stage 4 (Test) has never run, so there is not a single
measured metric or confusion matrix anywhere in the output tree.  The Results tab is
empty for the same reason.  Confusion matrices cannot be conjured from the weights:
the model has to be run over the held-out split and the predictions scored.

This script does exactly that, for every (model, task) whose weights exist, then
writes the workbooks.  It is resumable - anything already evaluated is skipped - so
you can stop it and restart it.

WHAT IT WRITES
--------------
    <out>/EXPORT/tasks_comparison_with_metrics.xlsx
        ALL_MODELS + one sheet per model + Dataset + PerClass_CI + Provenance
    <out>/EXPORT/confusion_matrices.xlsx
        one sheet per (model, task), counts and row percentages
    <out>/EXPORT/confusion_matrices/<model>__<task>.png
        rendered heat map for every pair
    <out>/EXPORT/per_class_metrics.xlsx
        per-class precision / recall / F1 / support with Wilson 95% CIs
    <out>/EXPORT/PREDICTIONS_FOR_KAPPA.csv
        image, side, reference grade, predicted grade, confidence, box
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import opg_metrics as M
import opg_scan as S
import opg_tasks as T


# --------------------------------------------------------------------------------------
def render_confusion(cm: pd.DataFrame, title: str, path: Path, note: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = cm.to_numpy(dtype=float)
    known = ~np.isnan(mat)
    rows = np.nansum(np.where(known, mat, 0), axis=1)
    pct = np.divide(np.where(known, mat, 0) * 100,
                    np.clip(rows[:, None], 1e-9, None))

    fig, ax = plt.subplots(figsize=(1.9 + 1.35 * mat.shape[1],
                                    1.9 + 1.15 * mat.shape[0]), dpi=160)
    ax.imshow(np.where(known, pct, np.nan), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(mat.shape[1]),
                  [str(c).replace("pred_", "") for c in cm.columns], rotation=30,
                  ha="right")
    ax.set_yticks(range(mat.shape[0]),
                  [str(i).replace("true_", "") for i in cm.index])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if known[i, j]:
                ax.text(j, i, f"{int(mat[i, j])}\n{pct[i, j]:.1f}%", ha="center",
                        va="center", fontsize=8.5,
                        color="white" if pct[i, j] > 55 else "black")
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True,
                                           facecolor="#eeeeee", hatch="//",
                                           edgecolor="#bbbbbb", lw=0))
                ax.text(j, i, "?", ha="center", va="center", fontsize=13,
                        color="#999999")
    if note:
        fig.text(0.5, 0.005, note[:120], ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion_from_cache(cache, task, conf) -> pd.DataFrame:
    spec = T.TASKS[task]
    if spec["kind"] == "classify":
        yt = np.array([c["gt"] for c in cache])
        yp = np.array([c["pred"] for c in cache])
        return M.confusion_df(yt, yp, include_missed=False)
    yt, yp, _, _ = M.object_level_pairs(cache, conf=conf)
    if spec["class_aware"]:
        return M.confusion_df(yt, yp, include_missed=True)
    # class-agnostic detector: grade x detected/missed
    return pd.DataFrame()


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser("export every metric and confusion matrix")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf-mode", default="auto", choices=["auto", "fixed"])
    ap.add_argument("--conf-eval", type=float, default=0.25)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="re-evaluate even if done")
    ap.add_argument("--models", default="", help="comma list; blank = every model found")
    ap.add_argument("--tasks", default="", help="comma list; blank = every trained task")
    return ap


def run(a, log=None):
    """Callable from the GUI as well as the command line."""
    out = Path(a.out)
    exp = out / "EXPORT"
    exp.mkdir(parents=True, exist_ok=True)
    fh = open(exp / "export_log.txt", "a", encoding="utf-8")
    _outer = log

    def log(m=""):
        if _outer:
            _outer(m)
        else:
            print(m, flush=True)
        fh.write(str(m) + "\n")
        fh.flush()

    log("=" * 78)
    log("STEP 1  discover what exists")
    log("=" * 78)
    disc = S.scan(a.root, out)
    for line in disc.summary_lines():
        log("  " + line)

    pairs = []
    want_models = [m.strip() for m in a.models.split(",") if m.strip()]
    want_tasks = [int(t) for t in a.tasks.split(",") if t.strip()]
    for model, per_task in disc.weights.items():
        if want_models and model not in want_models:
            continue
        for tid in sorted(per_task):
            if want_tasks and tid not in want_tasks:
                continue
            pairs.append((model, tid))
    # cascades need their two parents, not their own weights
    for model, per_task in disc.weights.items():
        if want_models and model not in want_models:
            continue
        for tid, spec in T.TASKS.items():
            if spec["kind"] != "cascade":
                continue
            if want_tasks and tid not in want_tasks:
                continue
            if spec["detector"] in per_task and spec["classifier"] in per_task:
                pairs.append((model, tid))
    pairs = sorted(set(pairs))

    if not pairs:
        log("\nNo trained weights found, so there is nothing to evaluate.")
        log("Train something first:  python opg_run.py --root ... --out ... "
            "--stages prepare,train")
        fh.close()
        return

    log(f"\n  {len(pairs)} (model, task) pairs have weights:")
    for m, t in pairs:
        log(f"    {m:10s} task {t}  {T.TASKS[t]['name']}")

    cfg = T.TrainConfig(device=a.device, conf_mode=a.conf_mode, conf_eval=a.conf_eval)
    pipe = T.Pipeline(out / "datasets", out / "results", cfg, log=log,
                      n_boot=a.bootstrap)

    log("\n" + "=" * 78)
    log("STEP 2  evaluate on the held-out test split")
    log("=" * 78)
    done, failed = [], []
    for model, tid in pairs:
        try:
            if pipe.is_evaluated(model, tid) and not a.force:
                log(f"[{model}] task {tid}: already evaluated - skipped")
                done.append((model, tid))
                continue
            pipe.test(model, tid, force=a.force)
            done.append((model, tid))
        except Exception as e:  # noqa: BLE001
            log(f"!! [{model}] task {tid} failed: {e}")
            log(traceback.format_exc(limit=3))
            failed.append((model, tid, str(e)))

    if not done:
        log("\nNothing evaluated successfully; no workbooks written.")
        fh.close()
        return

    log("\n" + "=" * 78)
    log("STEP 3  metrics workbook")
    log("=" * 78)
    models = sorted({m for m, _ in done})
    tasks = sorted({t for _, t in done})
    sizes = {}
    for m in models:
        try:
            sizes[m] = T.model_size_info(T.resolve_weights(m, "detect", log))
        except Exception:  # noqa: BLE001
            sizes[m] = {}
    wb = T.build_comparison_workbook(
        out / "results", models, tasks, exp / "tasks_comparison_with_metrics.xlsx",
        dataset_stats=(pd.read_csv(out / "datasets" / "dataset_statistics.csv")
                       if (out / "datasets" / "dataset_statistics.csv").exists()
                       else None),
        sizes=sizes, log=log)

    log("\n" + "=" * 78)
    log("STEP 4  confusion matrices, one per model and task")
    log("=" * 78)
    cm_sheets, cm_index = {}, []
    fig_dir = exp / "confusion_matrices"
    for model, tid in done:
        d = pipe.task_dir(model, tid)
        f = d / "tables" / "confusion_matrix.csv"
        if not f.exists():
            log(f"  [{model}] task {tid}: no confusion matrix produced "
                f"(class-agnostic task with no grade map)")
            continue
        cm = pd.read_csv(f, index_col=0)
        sheet = f"{model}_t{tid}"[:31]
        cm_sheets[sheet] = cm
        title = f"{model} - {T.TASKS[tid]['name']}"
        png = fig_dir / f"{model}__task{tid}.png"
        try:
            render_confusion(cm, title, png)
        except Exception as e:  # noqa: BLE001
            log(f"  ! render failed for {title}: {e}")
        tot = float(np.nansum(cm.to_numpy(dtype=float)))
        diag = float(np.nansum(np.diag(cm.to_numpy(dtype=float))[:3]))
        cm_index.append({"model": model, "task": tid, "name": T.TASKS[tid]["name"],
                         "sheet": sheet, "figure": png.name,
                         "items": int(tot),
                         "on-diagonal": int(diag),
                         "overall agreement": round(diag / max(1e-9, tot), 4)})
        log(f"  {title}: {int(tot)} items, {int(diag)} on-diagonal -> {png.name}")

    if cm_sheets:
        with pd.ExcelWriter(exp / "confusion_matrices.xlsx") as w:
            pd.DataFrame(cm_index).to_excel(w, sheet_name="INDEX", index=False)
            for name, cm in cm_sheets.items():
                cm.to_excel(w, sheet_name=name)
                mat = cm.to_numpy(dtype=float)
                rows = np.nansum(mat, axis=1, keepdims=True)
                pct = pd.DataFrame(np.round(mat / np.clip(rows, 1e-9, None) * 100, 2),
                                   index=cm.index, columns=cm.columns)
                pct.to_excel(w, sheet_name=name, startrow=len(cm) + 3)
        log(f"  wrote {exp / 'confusion_matrices.xlsx'} ({len(cm_sheets)} sheets)")

    log("\n" + "=" * 78)
    log("STEP 5  per-class metrics with Wilson 95% CIs")
    log("=" * 78)
    ci_frames = []
    for model, tid in done:
        try:
            ci = S.per_class_ci_table(disc, model, tid)
            if len(ci):
                ci.insert(0, "task", tid)
                ci.insert(0, "model", model)
                ci_frames.append(ci)
        except Exception as e:  # noqa: BLE001
            log(f"  ! CI failed for {model} task {tid}: {e}")
    if ci_frames:
        ci_all = pd.concat(ci_frames, ignore_index=True)
        ci_all.to_excel(exp / "per_class_metrics.xlsx", index=False)
        log(f"  wrote {exp / 'per_class_metrics.xlsx'} ({len(ci_all)} rows)")
        thin = ci_all[pd.to_numeric(ci_all["support (n)"], errors="coerce") <= 5]
        if len(thin):
            log(f"  ! {len(thin)} class rows have support <= 5. Their intervals span "
                f"most of [0,1]; report the count, not the point estimate.")

    log("\n" + "=" * 78)
    log("STEP 6  prediction list for the manual Kappa re-reading")
    log("=" * 78)
    preds = []
    for model, tid in done:
        f = pipe.task_dir(model, tid) / "predictions.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.insert(0, "Task", tid)
            df.insert(0, "Model", model)
            preds.append(df)
    if preds:
        allp = pd.concat(preds, ignore_index=True)
        allp.to_csv(exp / "PREDICTIONS_FOR_KAPPA.csv", index=False)
        log(f"  wrote {exp / 'PREDICTIONS_FOR_KAPPA.csv'} ({len(allp)} rows)")

    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"  evaluated : {len(done)} model/task pairs")
    if failed:
        log(f"  failed    : {len(failed)}")
        for m, t, e in failed:
            log(f"      {m} task {t}: {e[:90]}")
    log(f"  output    : {exp}")
    csv = exp / "tasks_comparison_with_metrics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        cols = [c for c in ["Model", "Task", "Task name", "Precision", "Recall", "F1",
                            "mAP@0.5", "Accuracy", "Operating threshold"] if c in df]
        log("\n" + df[cols].to_string(index=False))
    fh.close()


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
