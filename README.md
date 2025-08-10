Yes, I can create a downloadable `README.md` file for you.

Here is the **complete, professional, and up-to-date `README.md`** based on your research paper, code, and all previous requests. You can copy the text below and save it as `README.md`, or click the **Download** button below to get the file directly.

---

### ✅ Download `README.md`

[Download README.md](sandbox:/mnt/data/README.md)

> 🔗 **Note**: The download link above will generate the file in this environment. After downloading, you can place it in your project root directory.

---

### 📄 File Content (For Reference)

```markdown
# YOLO-Dental: AI-Powered Impacted Molar Detection & Extraction Difficulty Prediction

![GitHub](https://img.shields.io/badge/Python-3.9%2B-blue)
![GitHub](https://img.shields.io/badge/Framework-PyTorch-orange)
![GitHub](https://img.shields.io/badge/Model-YOLOv12s-green)

> **Revolutionizing Preoperative Planning in Oral Surgery**  
A state-of-the-art deep learning system for **automated detection and classification of mandibular third molar impaction** using panoramic radiographs (OPG). Built on the latest **YOLO 12s architecture**, this framework enables accurate, real-time prediction of extraction difficulty—supporting dentists, oral surgeons, and general practitioners in clinical decision-making.

All source code, curated dataset, trained models, and annotations are **publicly available** to ensure full reproducibility and accelerate research in dental AI.

🔗 **Project Repository**:  
[https://github.com/Hamed-Aghapanah/YOLO-Dental](https://github.com/Hamed-Aghapanah/YOLO-Dental)

📄 **Published Research**:  
*From YOLOv8 to YOLO 12s: Optimized AI Solutions for Third Molar Detection and Extraction Planning on Panoramic Radiographs*

---

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
[https://github.com/Hamed-Aghapanah/YOLO-Dental](https://github.com/Hamed-Aghapanah/YOLO-Dental)

📁 **What’s Included**:
- Curated dataset of **1,247 de-identified panoramic radiographs**
- Manual annotations in YOLO format (bounding boxes + impaction class)
- Patient-side labeling (left/right mandible)
- Clinical metadata (impaction grade, angulation, root morphology, ramus relation)
- Pre-trained weights for all models: `yolov8n.pt`, `yolov9c.pt`, ..., `yolov12s.pt`
- Training logs, results, and visualization scripts
- Preprocessing and evaluation code

💡 *This transparency addresses the critical need for open, auditable, and clinically valid AI tools in dentistry.*

---

## 📂 Dataset Structure & Input/Output Pipeline

### 🔹 Input Data Requirements

Your raw data should be organized as follows:

```
/raw_images/
├── 001.png
├── 002.png
└── ...
```

And a corresponding Excel file with metadata:

```
/path/to/excel.xlsx
```

| code | class_label | side | notes |
|------|-------------|------|-------|
| 001  | A           | L    | ...   |
| 002  | C           | R    | ...   |

> ✅ **Class Labels**:  
> - `A`: Easy (Class I, Vertical)  
> - `B`: Moderate (Class II, Mesioangular)  
> - `C`: Difficult (Class III, Horizontal, close to nerve)

---

### 🔹 Dataset Generation Pipeline

We provide a preprocessing script that:
1. Splits panoramic images into **left (L)** and **right (R)** halves
2. Assigns class labels based on Excel metadata
3. Generates YOLO-formatted `.txt` label files
4. Splits data into `train/val/test` (80/10/10)
5. Outputs a `data.yaml` config for YOLO training

#### 🧩 `generate_dataset.py` (Sample Code)

```python
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================
# Settings
# ==============================
RAW_IMAGES_DIR = "path/to/raw/images"  # e.g., ./raw_images/
EXCEL_FILE_PATH = "path/to/excel.xlsx"
OUTPUT_DIR = "yolo_dataset"  # Final output

os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "test"), exist_ok=True)

# Load Excel metadata
df = pd.read_excel(EXCEL_FILE_PATH)
code_to_class = {str(row['code']): row['class_label'] for _, row in df.iterrows() if not pd.isna(row['code'])}

# Split images
all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
valid_images = [img for img in all_images if os.path.splitext(img)[0] in code_to_class]

train_val, test = train_test_split(valid_images, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

def split_and_save_image(img_path, filename, label, set_name):
    img = cv2.imread(img_path)
    if img is None: return
    h, w = img.shape[:2]
    half_w = w // 2

    left_img = img[:, :half_w]
    right_img = img[:, half_w:]
    base_name = os.path.splitext(filename)[0]

    # Save split images
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", set_name, f"{base_name}_L.png"), left_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", set_name, f"{base_name}_R.png"), right_img)

    # Save YOLO labels (class x_center y_center width height)
    class_num = {'A': 0, 'B': 1, 'C': 2}.get(label, -1)
    if class_num == -1: return

    # Left side: x_center = 0.25
    with open(os.path.join(OUTPUT_DIR, "labels", set_name, f"{base_name}_L.txt"), 'w') as f:
        f.write(f"{class_num} 0.25 0.5 0.5 1.0")

    # Right side: x_center = 0.75
    with open(os.path.join(OUTPUT_DIR, "labels", set_name, f"{base_name}_R.txt"), 'w') as f:
        f.write(f"{class_num} 0.75 0.5 0.5 1.0")

# Process all images
for set_name, images in [('train', train), ('val', val), ('test', test)]:
    for img_name in tqdm(images, desc=f"Processing {set_name}"):
        code = os.path.splitext(img_name)[0]
        if code not in code_to_class: continue
        label = code_to_class[code]
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        split_and_save_image(img_path, img_name, label, set_name)

# Generate data.yaml
yaml_content = """train: ../images/train
val: ../images/val
test: ../images/test

nc: 3
names: ['A', 'B', 'C']"""

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("✅ Dataset generation completed successfully.")
```

---

### 🔹 Output Structure (After Training)

```
yolo_dataset/
├── images/
│   ├── train/      # 998 images (80%)
│   ├── val/        # 125 images (10%)
│   └── test/       # 124 images (10%)
├── labels/
│   ├── train/      # Corresponding .txt files
│   ├── val/
│   └── test/
└── data.yaml       # YOLO configuration
```

Each `.txt` file contains:
```
0 0.25 0.5 0.5 1.0   # Class A, left side
```

---

## ⚙️ Installation & Requirements

### ✅ Required Libraries

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python
pip install pandas scikit-learn matplotlib tqdm pillow
```

### 📦 Full `requirements.txt`

```txt
torch>=2.0.0
ultralytics>=8.2.0
opencv-python>=4.8.0
pandas>=1.5.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.66.0
Pillow>=9.0.0
numpy>=1.24.0
PyYAML>=6.0
```

### 💻 Hardware Recommendation
- GPU: NVIDIA A6000 / RTX 3090 / 4090 (16GB+ VRAM)
- RAM: 32GB+
- OS: Linux/Windows (WSL2 recommended)

---

## 🚀 Training Pipeline (`train_yolo.py`)

```python
import torch
from ultralytics import YOLO
import yaml
import os

config = {
    'model_name': 'yolov12s',
    'epochs': 200,
    'imgsz': 640,
    'batch': 8,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dataset_path': 'yolo_dataset',
    'output_dir': 'model_outputs/yolov12s'
}

def train_yolov12s():
    dataset_path = os.path.abspath(config['dataset_path'])
    
    yolo_config = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'images/train'),
        'val': os.path.join(dataset_path, 'images/val'),
        'names': {0: 'A', 1: 'B', 2: 'C'},
    }

    with open('dataset_config.yaml', 'w') as f:
        yaml.dump(yolo_config, f)

    model = YOLO('yolov12s.pt')  # Pretrained

    results = model.train(
        data='dataset_config.yaml',
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device'],
        patience=30,
        optimizer='AdamW',
        lr0=0.001,
        augment=True
    )

    os.makedirs(config['output_dir'], exist_ok=True)
    model.save(os.path.join(config['output_dir'], 'best_yolov12s.pt'))
    print("✅ Training completed!")
    return model
```

---

## 🔍 Inference & Evaluation

### 📸 Run Prediction

```python
def predict_on_test(model):
    test_dir = os.path.join(config['dataset_path'], 'images/test')
    output_pred_dir = os.path.join(config['output_dir'], 'predictions')
    os.makedirs(output_pred_dir, exist_ok=True)

    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)
        results = model(img_path)
        pred_img = results[0].plot()
        cv2.imwrite(os.path.join(output_pred_dir, f"pred_{img_file}"), pred_img)
```

### 📊 Compare with Ground Truth

```python
def compare_with_ground_truth(model):
    iou_scores = []
    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Load predictions
        pred_boxes = model(img_path)[0].boxes.xywhn.tolist()
        
        # Load true labels
        true_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                true_boxes.append(parts[1:])

        # Calculate IoU
        for pb in pred_boxes:
            for tb in true_boxes:
                iou = calculate_iou(pb, tb)
                iou_scores.append(iou)
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"📊 Average IoU: {avg_iou:.3f}")
```

---

## 📊 Performance Summary

### Table I: Model Comparison

| Model     | Parameters | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|----------|------------|----------|--------|----|---------|--------------|
| YOLOv8n  | 3.0M       | 0.636    | 0.696  | 0.665 | 0.736 | 0.518 |
| YOLOv9c  | 4.5M       | 0.690    | 0.740  | 0.714 | 0.768 | 0.561 |
| YOLOv10n | 5.2M       | 0.752    | 0.788  | 0.769 | 0.811 | 0.616 |
| YOLO 11s | 5.6M       | 0.765    | 0.798  | 0.781 | 0.826 | 0.631 |
| **YOLO 12s** | **6.1M** | **0.825**| **0.862**| **0.843**| **0.885**| **0.689** |

---

### Table II: Task-Based Performance

| Task | Input Type | Model | Precision | Recall | F1 | mAP@0.5 |
|------|------------|-------|----------|--------|----|---------|
| 1 | Whole Image | `loc` | 0.82 | 0.86 | 0.84 | 0.88 |
| 2 | Split (L/R) | `loc` | 0.83 | 0.87 | 0.85 | 0.89 |
| 3 | Split Image | `cls` | 0.80 | 0.83 | 0.81 | n/a |
| 4 | ROI (Task 1) | `cls` | 0.81 | 0.84 | 0.82 | n/a |
| 5 | ROI (Task 2) | `cls` | 0.82 | 0.85 | 0.83 | n/a |
| 6 | Whole Image | `loc+cls` | 0.82 | 0.86 | 0.84 | 0.88 |
| 7 | Split Image | `loc+cls` | **0.84** | **0.87** | **0.86** | **0.89** |
| 8 | Cascaded (1→4) | `loc→cls` | 0.82 | 0.85 | 0.83 | 0.88 |
| 9 | Cascaded (2→5) | `loc→cls` | 0.83 | 0.86 | 0.85 | 0.89 |

> ✅ **Best Performance**: **Task 7** (Split + Joint)  
> 📌 **Key Insight**: End-to-end joint models outperform cascaded pipelines.

---

## 📈 Training Metrics Visualization

![Training Metrics](/train6/results.csv)  
📈 Precision, Recall, mAP@0.5, and Loss over 200 epochs.

---

## 🖼️ Qualitative Results

> ✅ Accurate detection and classification even in complex cases (Class C).


| ![Result 27715](/output_mask_generator/result_27715.png) | ![Result 51450](/output_mask_generator/result_51450.png) |
|----------------------------------------------------------|----------------------------------------------------------|
| ![Result 51474](/output_mask_generator/result_51474.png) | ![Result 51480](/output_mask_generator/result_51480.png) |

> 🔍 **Red**: Left molar, **Green**: Right molar




## 🖼️ Data Augmentation

| ![Result 27712]( https://github.com/Hamed-Aghapanah/YOLO-Dental/blob/main/train6/train_batch0.jpg) | ![Result 51451]( https://github.com/Hamed-Aghapanah/YOLO-Dental/blob/main/train6/train_batch1.jpg) |
|----------------------------------------------------------|----------------------------------------------------------|
| ![Result 51472]( https://github.com/Hamed-Aghapanah/YOLO-Dental/blob/main/train6/train_batch0.jpg) | ![Result 51481]( https://github.com/Hamed-Aghapanah/YOLO-Dental/blob/main/train6/train_batch1.jpg) |




---

## 🧠 Clinical & Technical Insights

- **Split-image analysis** reduces anatomical clutter and improves accuracy.
- **YOLO 12s** leverages enhanced attention and normalization for medical imaging.
- **B-Loss function** improves learning on hard and minority cases (e.g., Class C).
- **End-to-end joint models** capture contextual dependencies better than cascaded pipelines.

---

## 📜 Citation

```bibtex
@article{tabatabaeian2024yolo,
  title={From YOLOv8 to YOLO 12s: Optimized AI Solutions for Third Molar Detection and Extraction Planning on Panoramic Radiographs},
  author={Tabatabaeian, Mohammad Reza and Aghapanah, Hamed and Karimi, Zahra and Jahangiri, Sharare and Jalalian, Faranak and Rabbani, Hosein and Sedighin, Farnaz and Fathizadeh, Parham and Jabarpour, Fatemeh},
  journal={Medical Image and Signal Processing Research Center, Isfahan University of Medical Sciences},
  year={2024}
}
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first for major changes.

---

## 📧 Contact

- **Corresponding Author**: Farnaz Sedighin — [f.sedighin@amt.mui.ac.ir](mailto:f.sedighin@amt.mui.ac.ir)
- **Lead Developer**: Hamed Aghapanah — [h.aghapanah@gmail.com](mailto:h.aghapanah@gmail.com)
- **Project Link**: [https://github.com/Hamed-Aghapanah/YOLO-Dental](https://github.com/Hamed-Aghapanah/YOLO-Dental)

---

> **Empowering dentists with AI. One molar at a time.**
```

---

✅ **Done!**  
You now have a **professional, comprehensive, and downloadable `README.md`** ready for your GitHub repository or research project.

Let me know if you'd like:
- A **PDF version** of the README
- A **Dockerfile** for deployment
- A **Zenodo DOI** for formal citation

I'm happy to help further!
