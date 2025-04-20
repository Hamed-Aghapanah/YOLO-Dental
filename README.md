### YOLO-Dental: AI-Powered Impacted Molar Detection & Extraction Difficulty Prediction

:  

```markdown
# YOLO-Dental: AI-Powered Impacted Molar Detection & Extraction Difficulty Prediction

![GitHub](https://img.shields.io/badge/Python-3.9%2B-blue)
![GitHub](https://img.shields.io/badge/Framework-PyTorch-orange)
![GitHub](https://img.shields.io/badge/Model-YOLOv10n-green)

A deep learning-based system for classifying mandibular third molar impaction grades and predicting extraction difficulty using panoramic radiographic images. Built with YOLOv8, YOLOv9, and YOLOv10 models.

## 📌 Key Features
- **Automated Diagnosis**: Classifies impaction grades (A/B/C) based on Pell & Gregory and Winter classifications.
- **Real-Time Prediction**: Lightweight YOLO models (YOLOv8n, YOLOv9c, YOLOv10n) for high-speed inference.
- **Clinical Accuracy**: Achieves **0.811 mAP@0.5** (YOLOv10n) on a dataset of 2000 labeled images.
- **End-to-End Pipeline**: Includes data preprocessing, model training, and evaluation scripts.

## 🚀 Installation
```bash
git clone https://github.com/yourusername/YOLO-Dental.git
cd YOLO-Dental
pip install -r requirements.txt
```

## 📂 Dataset Structure
- **Images**: Panoramic radiographs in `./yolo_dataset/images/` (train/val/test splits).
- **Labels**: YOLO-formatted annotations in `./yolo_dataset/labels/`.
- **Class Mapping**: `A` (Simple), `B` (Moderate), `C` (Complex).

## 🛠️ Usage
1. **Train Models**:
```bash
python YOLOv_4_comparance.py
```

2. **Evaluate Performance**:
- Metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95.
- Results saved in `./model_comparison/model_comparison.xlsx`.

3. **Inference on New Images**:
```bash
python predict.py --model yolov10n.pt --source test_image.jpg
```

## 📊 Results
| Model    | mAP@0.5 | Precision | Recall | F1-Score |
|----------|---------|-----------|--------|----------|
| YOLOv8n  | 0.736   | 0.636     | 0.696  | 0.665    |
| YOLOv9c  | 0.768   | 0.690     | 0.740  | 0.714    |
| YOLOv10n | 0.811   | 0.752     | 0.788  | 0.769    |

## 📜 Citation
If you use this project in your research, please cite:
```bibtex
@article{tabatabaeian2024yolo,
  title={YOLO-Based Evaluation of Mandibular Third Molar Impaction Grades},
  author={Tabatabaeian, Mohammad Reza et al.},
  journal={Journal of Dental AI},
  year={2024}
}
```

## 🤝 Contributing
Pull requests are welcome! For major changes, open an issue first.

## 📧 Contact
- **Email**: your-email@example.com
- **Project Link**: https://github.com/yourusername/YOLO-Dental
```

---

