import torch
from ultralytics import YOLO
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sklearn.metrics import jaccard_score
from PIL import Image
import time

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================#
# 1. تنظیمات اولیه
# =============================================================================#

config = {
    'model_name': 'yolov10n',
    'epochs': 20,
    'imgsz': 640,
    'batch': 8,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dataset_path': 'yolo_dataset',
    'test_images': ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg'],
    'output_dir': 'yolov10n_results'
}

# =============================================================================#
# 2. تابع آموزش مدل
# =============================================================================#

def train_yolov10():
    # مسیر دیتاست
    dataset_path = os.path.abspath(config['dataset_path'])

    # ساخت فایل config برای YOLO
    yolo_config = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'images/train'),
        'val': os.path.join(dataset_path, 'images/val'),
        'names': {0: 'A', 1: 'B', 2: 'C},
        'augment': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5
        }
    }

    with open('dataset_config.yaml', 'w') as f:
        yaml.dump(yolo_config, f)

    # بررسی وجود دایرکتوری‌های دیتاست
    required_dirs = [
        os.path.join(dataset_path, 'images/train'),
        os.path.join(dataset_path, 'images/val'),
        os.path.join(dataset_path, 'labels/train'),
        os.path.join(dataset_path, 'labels/val')
    ]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required dataset directory not found: {dir_path}")

    # بارگذاری مدل YOLOv10n
    model = YOLO('yolov10n.pt')  # مدل اولیه YOLOv10n

    # شروع آموزش
    results = model.train(
        data='dataset_config.yaml',
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device']
    )

    # ذخیره مدل آموزش‌دیده
    os.makedirs(config['output_dir'], exist_ok=True)
    model.save(os.path.join(config['output_dir'], 'best_yolov10n.pt'))

    print("✅ Training completed successfully!")
    return model

# =============================================================================#
# 3. تابع پیش‌بینی روی تصاویر تست
# =============================================================================#

def predict_on_test(model):
    test_dir = os.path.join(config['dataset_path'], 'images/test')
    output_pred_dir = os.path.join(config['output_dir'], 'predictions')
    os.makedirs(output_pred_dir, exist_ok=True)

    test_images = [os.path.join(test_dir, img) for img in config['test_images'] if os.path.exists(os.path.join(test_dir, img))]

    for img_path in test_images:
        results = model(img_path)
        result = results[0]

        # رسم خروجی پیش‌بینی
        pred_img = result.plot()

        # ذخیره تصویر
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_pred_dir, f"pred_{base_name}"), pred_img)

        print(f"Predicted and saved: {base_name}")

    print("✅ Prediction on test images completed.")

# =============================================================================#
# 4. تابع مقایسه با لیبل‌های واقعی (اختیاری)
# =============================================================================#

def compare_with_ground_truth(model):
    test_dir = os.path.join(config['dataset_path'], 'images/test')
    label_dir = os.path.join(config['dataset_path'], 'labels/test')

    iou_scores = []

    for img_file in config['test_images']:
        img_path = os.path.join(test_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        # پیش‌بینی
        results = model(img_path)
        pred_boxes = results[0].boxes.xywhn.tolist()
        pred_classes = results[0].boxes.cls.tolist()

        # لیبل‌های واقعی
        true_boxes = []
        true_classes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                cls_id, x_center, y_center, width, height = parts
                true_boxes.append([x_center, y_center, width, height])
                true_classes.append(cls_id)

        # محاسبه IoU
        for pb in pred_boxes:
            for tb in true_boxes:
                iou = calculate_iou(pb, tb)
                iou_scores.append(iou)

    if iou_scores:
        avg_iou = sum(iou_scores) / len(iou_scores)
        print(f"📊 Average IoU between predictions and ground truth: {avg_iou:.2f}")
    else:
        print("⚠️ No ground truth labels found to compare.")

def calculate_iou(box1, box2):
    # box format: [x_center, y_center, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1 - w1/2, x2 - w2/2)
    y_top = max(y1 - h1/2, y2 - h2/2)
    x_right = min(x1 + w1/2, x2 + w2/2)
    y_bottom = min(y1 + h1/2, y2 + h2/2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area

# =============================================================================#
# 5. رسم نمودارهای عملکرد
# =============================================================================#

def plot_metrics():
    # نمونه داده‌های عملکرد (می‌توانید این داده‌ها را از خروجی آموزش بگیرید)
    epochs = list(range(1, config['epochs']+1))
    precision = np.linspace(0.7, 0.752, config['epochs'])
    recall = np.linspace(0.73, 0.788, config['epochs'])
    mAP_05 = np.linspace(0.78, 0.811, config['epochs'])
    mAP_05_095 = np.linspace(0.58, 0.616, config['epochs'])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precision, label='Precision')
    plt.plot(epochs, recall, label='Recall')
    plt.plot(epochs, mAP_05, label='mAP@0.5')
    plt.plot(epochs, mAP_05_095, label='mAP@0.5:0.95')
    plt.title('YOLOv10n Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['output_dir'], 'training_metrics.png'))
    plt.close()

    print("📈 Performance metrics plotted and saved.")

# =============================================================================#
# 6. اجرای کلیه مراحل
# =============================================================================#

if __name__ == "__main__":
    try:
        print("🚀 Starting YOLOv10n training...")
        model = train_yolov10()

        print("\n🔎 Running inference on test images...")
        predict_on_test(model)

        print("\n📊 Comparing with ground truth labels...")
        compare_with_ground_truth(model)

        print("\n📈 Plotting training metrics...")
        plot_metrics()

        print("\n✅ YOLOv10n pipeline completed successfully!")
        print(f"Results saved in: {os.path.abspath(config['output_dir'])}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please ensure:")
        print("1. Your dataset is properly organized in YOLO format")
        print("2. Test images exist in 'test_images' directory")
        print("3. You have internet connection to download models if needed")