import torch
from ultralytics import YOLO
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================#
# 1. تنظیمات اولیه
# =============================================================================#

config = {
    'model_name': 'yolov10n',
    'epochs': 20,
    'imgsz': 224,
    'batch': 16,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dataset_path': 'Task5_WITH_LOCAL_CLASSIFICATION',
    'output_dir': 'yolov10n_Task5_results'
}

# =============================================================================#
# 2. تابع آموزش مدل
# =============================================================================#

def train_yolov10_task5():
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

    with open('dataset_config_task5.yaml', 'w') as f:
        yaml.dump(yolo_config, f)

    # بررسی وجود دایرکتوری‌های دیتاست
    required_dirs = [
        os.path.join(dataset_path, 'images/train'),
        os.path.join(dataset_path, 'images/val')
    ]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required dataset directory not found: {dir_path}")

    # بارگذاری مدل YOLOv10n
    model = YOLO('yolov10n.pt')  # مدل اولیه YOLOv10n

    # شروع آموزش
    results = model.train(
        data='dataset_config_task5.yaml',
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device'],
        project=config['output_dir'],
        name='classification',
        exist_ok=True
    )

    print("✅ Training completed successfully!")
    return model

# =============================================================================#
# 3. اجرای آموزش
# =============================================================================#

if __name__ == "__main__":
    try:
        print("🚀 Starting YOLOv10n training for Task5...")
        model = train_yolov10_task5()

        print("\n📈 Plotting training metrics...")
        plt.plot(results.metrics.train.loss, label='Train Loss')
        plt.plot(results.metrics.val.loss, label='Validation Loss')
        plt.title('YOLOv10n Training Loss - Task5')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config['output_dir'], 'loss_curve.png'))
        plt.close()

        print("\n✅ YOLOv10n pipeline for Task5 completed successfully!")
        print(f"Results saved in: {os.path.abspath(config['output_dir'])}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please ensure:")
        print("1. Your dataset is properly organized in YOLO format")
        print("2. You have internet connection to download models if needed")