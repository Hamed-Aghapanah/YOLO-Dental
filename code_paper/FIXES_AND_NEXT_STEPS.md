# What your first real run showed, and what changed

## 1. The exit errors are fixed

```
invalid command name "2956256803008tick"   ("after" script)
invalid command name "...update"  "...check_dpi_scaling"  "..._drain"
```

Repeating `after()` callbacks — the animation tweens, the log drain, the status
refresh — kept firing while Tk was tearing the window down, so each one addressed a
widget that no longer existed. Harmless, but it looks like a crash.

Both GUIs now cancel every pending timer on close. `Animator.shutdown()` disables
further scheduling (distinct from `cancel_all()`, which is used on theme change and
must leave animation enabled), and each window registers `WM_DELETE_WINDOW`.

## 2. Your real dataset is not what the manuscript says

Straight from your own scan of `manifest.xlsx`:

| split | images | A | B | C | molars |
|---|---|---|---|---|---|
| train | 676 | 603 | 489 | 12 | 1104 |
| val | 85 | 84 | 60 | 3 | 147 |
| test | 83 | 70 | 66 | 1 | 137 |
| **TOTAL** | **844** | **757** | **615** | **16** | **1388** |

The manuscript claims **1,247 images, A 790 / B 446 / C 11, test n = 124**.

Neither number is wrong exactly — they count different things. 1,247 is the raw
archive; 844 is the count of images that have **both** a grade in `class.xlsx` **and**
a box in `localize.csv`. The other ~400 are missing one or the other, so they cannot
train a detector. 1,388 is molars, not images.

Three consequences for the paper:

- The dataset section must state the matched count and say how the difference arose.
  `manifest.xlsx` → `UNMATCHED` sheet lists every excluded case with its reason.
- The class distribution changes materially: **B is 44%, not 36%**. The "extreme
  imbalance" framing still holds for C, but the A:B ratio in the paper is wrong.
- **Class C in test = 1**, now confirmed independently from your own data rather than
  from arithmetic. That settles the 5-versus-1 inconsistency for good.

## 3. There are no measured results yet

Your Metrics tab reads **"0 measured · 13 from the manuscript"**, and the Results tab
is empty. Eight checkpoints exist under `results/`, but stage 4 (Test) has never run,
so nothing has been scored on the held-out split.

This is why request 1 could not simply be "generate the confusion matrices": a
confusion matrix is not stored in a `.pt`. The model has to be run over the test split
and the predictions matched to the reference boxes.

```bash
python opg_export_all.py --root E:\project\0000_OPG\203-opg ^
                         --out  E:\project\0000_OPG\203-opg\output
```

or press **Analysis → Export EVERYTHING to Excel** in the GUI. It evaluates every
(model, task) that has weights, skips anything already done, and writes:

```
<output>/EXPORT/tasks_comparison_with_metrics.xlsx   ALL_MODELS + one sheet per model
<output>/EXPORT/confusion_matrices.xlsx              one sheet per (model, task)
<output>/EXPORT/confusion_matrices/*.png             rendered heat map for each pair
<output>/EXPORT/per_class_metrics.xlsx               per-class P/R/F1 + Wilson 95% CIs
<output>/EXPORT/PREDICTIONS_FOR_KAPPA.csv            for Dr. Karimi
```

## 4. One GUI

`opg_gui.py` gains an **Analysis** tab: it opens the six dashboard tabs as a second
window sharing the same event loop, with the project and output paths already filled
in and the scan started. Training and analysis are now one program.

## 5. Two things in the Test tab worth reading

Your screenshot of `27666.png` shows **10 detections** on an image that has at most
two molars, with confidences from 0.622 down to 0.313. That is the low-confidence
junk the F1 curve already warned about; the auto operating threshold (about 0.50 on
task 1) removes it. Judge detections at the operating point, not at 0.25.

The heat map reported *"detection-derived Gaussian (NOT network attention)"*, meaning
the Eigen-CAM hook did not attach to that checkpoint and it fell back to drawing the
output. Usable as an illustration, but it is **not** evidence of what the network
attends to, and the stamp on the image says so. Do not label it Grad-CAM in a figure.
