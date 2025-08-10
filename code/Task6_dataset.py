import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==============================
# تنظیمات
# ==============================
RAW_IMAGES_DIR = "path/to/raw/images"  # تصاویر اصلی نیم‌صفحه (تصاویر Task4)
YOLO_LABELS_DIR = "path/to/yolo_labels_Task3"  # خروجی Task3
EXCEL_FILE_PATH = "path/to/excel.xlsx"  # شامل برچسب A/B/C
OUTPUT_DIR = "Task6_LOCALIZER_TO_CLASSIFIER_PIPELINE"
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "test"), exist_ok=True)

# ==============================
# بارگذاری لیبل‌ها
# ==============================
df = pd.read_excel(EXCEL_FILE_PATH)
code_to_class = {}
for _, row in df.iterrows():
    code = str(row['code']) if not pd.isna(row['code']) else None
    class_label = row['class_label'] if not pd.isna(row['class_label']) else None
    if code and class_label:
        code_to_class[code] = class_label

# تقسیم تصاویر
all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(".png")]
valid_images = [img for img in all_images if os.path.splitext(img)[0] in code_to_class]
train_val, test = train_test_split(valid_images, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

# ==============================
# تابع برای Crop
# ==============================
def process_and_save_image(img_path, filename, output_dir, label, set_name):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    h, w = img.shape[:2]
    base_name = os.path.splitext(filename)[0]

    # خواندن BBox از Task3
    label_file = os.path.join(YOLO_LABELS_DIR, f"{base_name}.txt")
    if not os.path.exists(label_file):
        print(f"No label file found for {filename}")
        return

    with open(label_file, 'r') as f:
        line = f.readline()
        if not line:
            return
        parts = list(map(float, line.strip().split()))
        cls_id, x_center, y_center, width, height = parts

    # محاسبه پیکسلی BBox
    x_min = int((x_center - width / 2) * w)
    y_min = int((y_center - height / 2) * h)
    x_max = int((x_center + width / 2) * w)
    y_max = int((y_center + height / 2) * h)

    x_min, y_min, x_max, y_max = max(x_min, 0), max(y_min, 0), min(x_max, w), min(y_max, h)

    cropped_img = img[y_min:y_max, x_min:x_max]
    if cropped_img.size == 0:
        print(f"Cropped image empty: {filename}")
        return

    # ذخیره تصویر
    cropped_filename = f"{base_name}_cropped_{label}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", set_name, cropped_filename), cropped_img)

# ==============================
# پردازش
# ==============================
for set_name, images in [('train', train), ('val', val), ('test', test)]:
    for img_name in tqdm(images, desc=f"Processing {set_name}"):
        code = os.path.splitext(img_name)[0]
        if code not in code_to_class:
            continue
        label = code_to_class[code]
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        process_and_save_image(img_path, img_name, OUTPUT_DIR, label, set_name)

print("✅ Dataset generation completed successfully.")