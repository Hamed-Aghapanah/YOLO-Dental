# YOLO-Dental: AI-Powered Impacted Molar Detection & Extraction Difficulty Prediction

> **Automated detection and grading of mandibular third molars on panoramic radiographs**

A deep-learning pipeline that localises each impacted mandibular third molar on an
orthopantomogram (OPG) and assigns it a surgical difficulty grade — **A (easy)**,
**B (moderate)** or **C (difficult)** — derived from the Pell & Gregory and Winter
classification systems. Five YOLO generations plus RT-DETR are benchmarked on one
cohort under a single protocol, across nine explicitly defined tasks.

🔗 **Repository**: [https://github.com/Hamed-Aghapanah/YOLO-Dental](https://github.com/Hamed-Aghapanah/YOLO-Dental)

📄 **Paper**: *From YOLOv8 to YOLOv12s: Optimized AI Solutions for Third Molar Detection
and Extraction Planning on Panoramic Radiographs*

---

## 🎬 The software

| Training & analysis pipeline | Inference and overlay explorer |
|---|---|
| ![Pipeline GUI](gif/pipeline.gif) | ![Test tab](gif/test_overlays.gif) |

`gif/pipeline.gif` — the six-stage workflow: scan, prepare, tune, train, test, summary,
with the live status table.
`gif/test_overlays.gif` — the Test tab: load any checkpoint, toggle heat map, mask,
contour and confidence layers independently.

---

## 📌 Key features

- **Nine tasks, one protocol** — whole-image and split-image localisation, frame and
  ROI classification, unified detection+grading, and two cascaded pipelines.
- **Eight model families** — YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8,
  YOLOv5u and **RT-DETR**, selectable per run.
- **Resumable** — finished training is skipped; an interrupted run continues from
  `last.pt`.
- **Honest evaluation** — precision/recall/F1 are reported at an operating point
  chosen on the *validation* split, never on test; per-class Wilson confidence
  intervals accompany every point estimate.
- **Two GUIs in one** — a training pipeline and a six-tab analysis dashboard, five
  themes, live charts.
- **Reviewer-ready outputs** — confusion matrices, ROC curves with AUC intervals,
  per-class metrics and a prediction list for manual Kappa, all exported to Excel.

---

## 📊 Dataset

The archive holds 1,247 de-identified panoramic radiographs. A radiograph enters model
development only if it carries **both** a difficulty grade **and** a delineated
bounding box:

| Partition | Radiographs | Molars | Class A | Class B | Class C |
|---|---|---|---|---|---|
| Training (80%) | 676 | 1,104 | 603 | 489 | 12 |
| Validation (10%) | 85 | 147 | 84 | 60 | 3 |
| Independent test (10%) | 83 | 137 | 70 | 66 | 1 |
| **Total** | **844** | **1,388** | **757** (54.5%) | **615** (44.3%) | **16** (1.2%) |

539 radiographs carry both molars, 305 a single side. Splitting is at the **radiograph**
level and stratified, so the two molars of one patient never straddle the train/test
boundary.

> ⚠️ **Class C is represented by one molar in the independent test set.** A Wilson 95%
> interval on a single observation spans `[0.207, 1.000]` whichever way it falls, so
> per-class Class C metrics are reported as **not estimable** rather than as 0 or 1.
> Nothing in this repository supports a claim about automated assessment of the most
> difficult extractions.

### Input format

```
<project>/
├── images/<code>.png            panoramic radiographs
├── class.xlsx                   code | right | left | impaction | ramus relation |
│                                angulation | root morphology | root curvture |
│                                degree of hardnes | class_label
└── localize.xlsx                label_name | bbox_x | bbox_y | bbox_width |
                                 bbox_height | image_name | image_width | image_height
```

Both sheets carry the **patient** side, so the join key is `(code, side)` — no
geometric guessing. Verified on case 62299: the localize rows normalise to
`cx = 0.695424` (label L) and `cx = 0.299153` (label R), matching `62299.txt` exactly.

```
label_name "L" = patient left  = image RIGHT half   (standard OPG display)
label_name "R" = patient right = image LEFT  half
```

Two quirks of `class.xlsx` are handled automatically: a duplicated block is removed,
and the side is read from the **cell value** when it is `L`/`R` rather than from the
column header, because `right`/`left` are swapped on some rows (code 62299 carries
`right='L'`). Under that rule the sheet has zero self-contradicting `(code, side)`
groups; under the naive rule it has 24.

---

## ⚖️ Pre-trained weights

All checkpoints live in `weights/`. Load any of them directly:

```python
from ultralytics import YOLO
model = YOLO("weights/yolo26n/task7_joint_split/best.pt")
results = model.predict("images/62299.png", conf=0.50)
```

```
weights/
├── yolo26n/  task1_det_whole/best.pt  task2_det_split/best.pt  task3_cls_split/best.pt
│             task4_cls_roi_whole/best.pt  task5_cls_roi_split/best.pt
│             task6_joint_whole/best.pt  task7_joint_split/best.pt
├── yolo26s/  … same nine task folders …
├── yolo12s/  …
├── yolo11s/  …
├── yolov10n/ yolov9c/ yolov8n/          Table II baselines (task 1 only)
└── rtdetr-l/                            transformer baseline
```

Each task folder also contains `last.pt` (for resuming) and `results.csv` (the real
training curve). Tasks 8 and 9 are cascades: they store no weights of their own and
reuse Task 1+4 and Task 2+5 respectively.

> Use the operating threshold recorded in `metrics.json` for each checkpoint, not the
> Ultralytics default of 0.25. On task 1 the F1-optimal threshold is ≈ 0.50; at 0.25
> precision falls to about 0.18 because the detector emits many low-confidence boxes.

---

## ⚙️ Installation

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas \
            matplotlib scikit-learn scipy openpyxl customtkinter pillow

python studio_selftest.py     # must end: ALL STUDIO TESTS PASSED
```

**GPU note.** A CUDA 11.8 build of torch does **not** support Blackwell cards
(RTX 5060, compute 12.0). For those install from the cu128 index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The Hardware tab detects this mismatch and warns before you waste a run.

---

## 🚀 Usage

```bash
python opg_gui.py                                    # pipeline GUI (6 tabs)
python opg_dashboard.py                              # analysis dashboard (6 tabs)

python opg_run.py --root <project> --out <output>    # full run, no GUI
python opg_run.py --root … --out … --stages status   # what is already finished
python opg_run.py --list-models                      # models + hardware profiles
python opg_export_all.py --root … --out …            # every metric + confusion matrix
```

Stages are independent and resumable: `prepare → tune → train → test → summary`.

### Hardware profiles

| profile | card | batch | imgsz | workers | epochs | cache |
|---|---|---|---|---|---|---|
| `gtx980m` | GTX 980M, 4 GB | 8 | 640 | 4 | 150 | off |
| `rtx3060_6` | RTX 3060 Laptop, 6 GB | 12 | 640 | 6 | 200 | disk |
| `rtx3060_12` | RTX 3060, 12 GB | 24 | 640 | 8 | 200 | ram |
| `rtx5060` | RTX 5060, 8 GB | 16 | 640 | 8 | 200 | ram |

Batch is scaled per architecture: ÷2 for `-m`/`-l`, ÷4 for RT-DETR.

---

## 📁 Outputs

```
<output>/
├── datasets/manifest.xlsx           ALL_MOLARS · BOTH_SIDES · SPLIT_SUMMARY · UNMATCHED
├── results/<model>/<task>/
│   ├── train/weights/best.pt · last.pt · results.csv
│   ├── figures/  confusion_matrix.png · roc_curves.png · training_curves.png
│   ├── tables/   per_class.csv · confusion_matrix.csv · auc.csv · bootstrap.csv
│   ├── predictions.csv · metrics.json
│   └── train_done.json · eval_done.json      ← drive the skip logic
└── EXPORT/
    ├── tasks_comparison_with_metrics.xlsx    one sheet per model
    ├── confusion_matrices.xlsx + PNGs        one per (model, task)
    ├── per_class_metrics.xlsx                Wilson 95% CIs
    └── PREDICTIONS_FOR_KAPPA.csv
```

---

## 🧠 Notes that affect what you may publish

- **"Dice" here is bounding-box Dice** — 2·overlap/(area₁+area₂) between predicted and
  reference *boxes*. There are no pixel masks in this dataset; it is not a
  segmentation Dice.
- **Heat maps** use Eigen-CAM from a backbone hook when the hook attaches; otherwise
  they fall back to a detection-derived Gaussian. The two are stamped differently on
  the image. Only the first shows network attention — do not label the fallback
  Grad-CAM.
- **Horizontal flip is disabled.** Mirroring an OPG swaps patient left and right,
  which is the distinction tasks 2/3/5/7 exist to learn.
- **Focal loss is not applied to RT-DETR**, which uses its own varifocal-style loss.
- **Human and model Kappa come from different samples** — the 167-case reliability
  subset versus the independent test partition — so that comparison is indicative,
  not an equivalence test.

---

## 📜 Citation

```bibtex
@article{aghapanah2026yolodental,
  title  = {From YOLOv8 to YOLOv12s: Optimized AI Solutions for Third Molar Detection
            and Extraction Planning on Panoramic Radiographs},
  author = {Aghapanah, Hamed and Tabatabaeian, Mohammad Reza and Karimi, Zahra and
            Jahangiri, Sharare and Jalalian, Faranak and Rabbani, Hossein and
            Sedighin, Farnaz and Fathizadeh, Parham and Jabarpour, Fatemeh},
  journal= {Medical Image and Signal Processing Research Center,
            Isfahan University of Medical Sciences},
  year   = {2026}
}
```

Ethics approval IR.MUI.DHMT.REC.1402.078 · Funded by Isfahan University of Medical
Sciences, grant No. 2402226.

## 📧 Contact

- **Corresponding author**: Farnaz Sedighin — f.sedighin@amt.mui.ac.ir
- **Lead developer**: Hamed Aghapanah — h.aghapanah@gmail.com
