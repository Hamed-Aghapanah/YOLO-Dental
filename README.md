# YOLO-Dental: AI-Powered Impacted Molar Detection & Extraction Difficulty Prediction

![GitHub](https://img.shields.io/badge/Python-3.9%2B-blue)
![GitHub](https://img.shields.io/badge/Framework-PyTorch-orange)
![GitHub](https://img.shields.io/badge/Model-YOLOv12s-green)

> **Revolutionizing Preoperative Planning in Oral Surgery**  
A state-of-the-art deep learning system for **automated detection and classification of mandibular third molar impaction** using panoramic radiographs (OPG). Built on the latest **YOLO 12s architecture**, this framework enables accurate, real-time prediction of extraction difficulty—supporting dentists, oral surgeons, and general practitioners in clinical decision-making.

## 📌 Key Features

- **Clinically Grounded Classification**: Classifies impaction severity into **Class A (Easy)**, **B (Moderate)**, and **C (Difficult)** using **Pell & Gregory** and **Winter classification systems**.
- **Advanced YOLO Pipeline**: Implements and benchmarks **YOLOv8n → YOLO 12s**, with **YOLO 12s** achieving **0.885 mAP@0.5** — the highest reported accuracy for this task.
- **Optimized Inference Strategy**: Supports **whole-image**, **split-image (left/right)**, and **cascaded vs. end-to-end** workflows. The **split + joint localization-classification** approach (Task 7) delivers best-in-class performance.
- **Real-World Clinical Design**: Trained on **1,247 real-world OPGs** with expert annotations, addressing class imbalance and anatomical variability.
- **Robust Training & Loss Design**: Uses **B-Loss function** for balanced learning across hard cases and minority classes (e.g., Class C).
- **Publicly Available**: All **code, trained models, dataset, and annotations** are open-sourced to promote reproducibility and clinical AI advancement.

---

## 🌐 Data & Model Availability

✅ **All resources are publicly available** to ensure full reproducibility and accelerate future research in dental AI.

🔗 **GitHub Repository**:  
[https://github.com/yourusername/YOLO-Dental](https://github.com/yourusername/YOLO-Dental)  
*(Replace with actual public link upon publication)*

📁 **What’s Included**:
- Curated dataset of **1,247 de-identified panoramic radiographs**
- Manual annotations in YOLO format (bounding boxes + impaction class)
- Patient-side labeling (left/right mandible)
- Clinical metadata: angulation, root morphology, ramus relation
- Pre-trained weights for all models: `yolov8n.pt`, `yolov9c.pt`, ..., `yolov12s.pt`
- Training logs, results, and visualization scripts

💡 *This transparency addresses the critical need for open, auditable, and clinically valid AI tools in dentistry.*

---

## 📊 Performance Summary

### Table I: Comparative Localization Performance of YOLO Models

| Model     | Parameters | Layers | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
|----------|------------|--------|----------|--------|----------|---------|--------------|
| YOLOv8n  | 3.0M       | 72     | 0.636    | 0.696  | 0.665    | 0.736   | 0.518        |
| YOLOv9c  | 4.5M       | 85     | 0.690    | 0.740  | 0.714    | 0.768   | 0.561        |
| YOLOv10n | 5.2M       | 90     | 0.752    | 0.788  | 0.769    | 0.811   | 0.616        |
| YOLO 11s | 5.6M       | 96     | 0.765    | 0.798  | 0.781    | 0.826   | 0.631        |
| **YOLO 12s** | **6.1M**   | **102** | **0.825**| **0.862**| **0.843**| **0.885**| **0.689**    |

> ✅ **YOLO 12s** outperforms all predecessors, showing the value of architectural refinement in **medical imaging**.

---

## 🔍 Multi-Stage Evaluation: Task-Based Performance

| Task | Input Type | Model Type | Description | Precision | Recall | F1 | mAP@0.5 |
|------|------------|------------|-------------|----------|--------|----|---------|
| 1 | Whole Image | `loc` | Full-image localization | 0.82 | 0.86 | 0.84 | 0.88 |
| 2 | Split (L/R) | `loc` | Left/right localization | 0.83 | 0.87 | 0.85 | 0.89 |
| 3 | Split Image | `cls` | Frame-level classification | 0.80 | 0.83 | 0.81 | n/a |
| 4 | ROI (from Task 1) | `cls` | Classification on cropped ROI | 0.81 | 0.84 | 0.82 | n/a |
| 5 | ROI (from Task 2) | `cls` | ROI classification (split) | 0.82 | 0.85 | 0.83 | n/a |
| 6 | Whole Image | `loc+cls` | Joint localization & classification | 0.82 | 0.86 | 0.84 | 0.88 |
| 7 | Split Image | `loc+cls` | **Best: Split + Joint** | **0.84** | **0.87** | **0.86** | **0.89** |
| 8 | Cascaded (Task1→4) | `loc→cls` | Two-stage: detect then classify | 0.82 | 0.85 | 0.83 | 0.88 |
| 9 | Cascaded (Task2→5) | `loc→cls` | Two-stage on split images | 0.83 | 0.86 | 0.85 | 0.89 |

> 📌 **Key Insight**:  
> - **Split-image analysis** improves performance by reducing anatomical clutter.
> - **End-to-end (joint)** models (Tasks 6 & 7) outperform **cascaded pipelines** (Tasks 8 & 9), as they learn contextual dependencies between localization and classification.
> - **Task 7** (split + joint) is optimal: **Precision: 0.84, mAP@0.5: 0.89**

---

## 📈 Training Dynamics & Convergence

### Training Progress (YOLO 12s - Task 7)

![Train Batch 0](/train6/train_batch0.jpg)  
*Early epoch: Model begins to localize molars with coarse accuracy.*

![Train Batch 1](/train6/train_batch1.jpg)  
*Mid-training: Bounding boxes tighten around teeth; class confidence improves.*

![Train Batch 2](/train6/train_batch2.jpg)  
*Late epoch: High-confidence, precise localization and correct classification.*

### Loss & Metric Trends

![Training Curves](/train6/results.csv)  
📈 **Training convergence** over 200 epochs (early stopping applied).  
- **Box loss**, **cls loss**, and **dfl loss** decrease steadily.
- **Precision**, **Recall**, and **mAP@0.5** plateau after ~150 epochs.
- No signs of overfitting due to **augmentation + B-Loss + early stopping**.

📎 **Correlogram**: `/train6/labels_correlogram.jpg` shows balanced label distribution and minimal class correlation issues.

---

## 🖼️ Qualitative Results: Real-World OPG Predictions

### Detected Impacted Molars (Red: Left, Green: Right)

| ![Result 27715](/output_mask_generator/result_27715.png) | ![Result 27721](/output_mask_generator/result_27721.png) | ![Result 27726](/output_mask_generator/result_27726.png) |
|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
| ![Result 51450](/output_mask_generator/result_51450.png) | ![Result 51455](/output_mask_generator/result_51455.png) | ![Result 51461](/output_mask_generator/result_51461.png) |
| ![Result 51474](/output_mask_generator/result_51474.png) | ![Result 51480](/output_mask_generator/result_51480.png) | ![Result 51491](/output_mask_generator/result_51491.png) |

> 🔍 **Visual Analysis**:
> - Model accurately localizes **single and bilateral impactions**.
> - Correctly classifies **Class A (easy)** vs. **Class C (difficult)** cases with high confidence.
> - Handles **low-contrast regions**, **overlapping roots**, and **partial occlusions**.
> - Side-aware detection (left/right) enables surgical planning per quadrant.

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/YOLO-Dental.git
cd YOLO-Dental
pip install -r requirements.txt
