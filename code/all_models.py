import torch
from ultralytics import YOLO

# لیست مدل‌های YOLO در سری‌های مختلف
yolo_models = [
    # YOLOv5 (نسخه‌های u)
    "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu",
    
    # YOLOv6 (ممکن است موجود نباشد)
    "yolov6n", "yolov6s", "yolov6m", "yolov6l", "yolov6t",

    # YOLOv7 (ممکن است موجود نباشد)
    "yolov7", "yolov7x", "yolov7tiny",

    # YOLOv8
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
    "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
    "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",

    # YOLOv9
    "yolov9c", "yolov9e",

    # YOLOv10
    "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",

    # YOLOv11 (ممکن است موجود نباشد)
    "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x",

    # YOLOv12 (ممکن است موجود نباشد)
    "yolov12n", "yolov12s", "yolov12m", "yolov12l", "yolov12x"
]

# تابع برای محاسبه تعداد پارامترها
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# تابع اصلی برای نمایش اطلاعات مدل‌ها
def display_yolo_models_info():
    print(f"{'Model Name':<15} | {'Parameters':<15} | {'Input Size':<15}")
    print("-" * 50)
    
    for model_name in yolo_models:
        try:
            # بارگذاری مدل
            model = YOLO(model_name + ".pt")
            
            # محاسبه تعداد پارامترها
            num_params = count_parameters(model.model)
            
            # دریافت اندازه ورودی مدل
            input_size = model.model.stride.max().item() * 32  # اندازه تصویر ورودی
            
            # نمایش اطلاعات
            print(f"{model_name:<15} | {num_params:<15,} | {input_size:<15}")
        except Exception as e:
            print(f"{model_name:<15} | {'Not Available':<15} | {str(e):<15}")

# فراخوانی تابع اصلی
display_yolo_models_info()