import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================
# تنظیمات
# ==============================
RAW_IMAGES_DIR = "path/to/raw/images"  # پوشه تصاویر اصلی
EXCEL_FILE_PATH = "path/to/excel.xlsx"  # فایل اکسل حاوی اطلاعات برچسب‌ها
OUTPUT_DIR = "Task2_localization_and_labeling"

# اطمینان از وجود پوشه‌های خروجی
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "test"), exist_ok=True)

# ==============================
# بارگذاری اطلاعات اکسل
# ==============================
df = pd.read_excel(EXCEL_FILE_PATH)
code_to_class = {}
for _, row in df.iterrows():
    code = str(row['code']) if not pd.isna(row['code']) else None
    class_label = row['class_label'] if not pd.isna(row['class_label']) else None
    if code and class_label:
        code_to_class[code] = class_label

# ==============================
# تقسیم تصاویر به Train / Val / Test
# ==============================
all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith('.png')]
valid_images = [img for img in all_images if os.path.splitext(img)[0] in code_to_class]

train_val, test = train_test_split(valid_images, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, random_state=42)

# ==============================
# تابع برای تقسیم تصویر و ذخیره نیم‌تصاویر
# ==============================
def split_and_save_image(img_path, filename, output_dir, label, set_name):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    h, w = img.shape[:2]
    half_w = w // 2

    left_img = img[:, :half_w]
    right_img = img[:, half_w:]

    base_name = os.path.splitext(filename)[0]

    # ذخیره نیم تصاویر
    left_filename = f"{base_name}_L_{label}.png"
    right_filename = f"{base_name}_R_{label}.png"

    left_label_file = f"{base_name}_L_{label}.txt"
    right_label_file = f"{base_name}_R_{label}.txt"

    # ذخیره تصاویر
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", set_name, left_filename), left_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", set_name, right_filename), right_img)

    # تبدیل برچسب به شماره کلاس
    class_num = {'A': 0, 'B': 1, 'C': 2}.get(label, -1)
    if class_num == -1:
        return

    # ذخیره لیبل (YOLO format)
    with open(os.path.join(OUTPUT_DIR, "labels", set_name, left_label_file), 'w') as f:
        x_center = 0.25  # چون نیم چپ است
        y_center = 0.5
        width = 0.5
        height = 1.0
        f.write(f"{class_num} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(os.path.join(OUTPUT_DIR, "labels", set_name, right_label_file), 'w') as f:
        x_center = 0.75
        y_center = 0.5
        width = 0.5
        height = 1.0
        f.write(f"{class_num} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

# ==============================
# پردازش تصاویر
# ==============================
for set_name, images in [('train', train), ('val', val), ('test', test)]:
    for img_name in tqdm(images, desc=f"Processing {set_name}"):
        code = os.path.splitext(img_name)[0]
        if code not in code_to_class:
            continue
        label = code_to_class[code]
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        split_and_save_image(img_path, img_name, OUTPUT_DIR, label, set_name)

# ==============================
# تولید فایل data.yaml
# ==============================
yaml_content = f"""train: ../images/train
val: ../images/val
test: ../images/test

nc: 3
names: ['A', 'B', 'C']"""

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("✅ Dataset generation completed successfully.")