import os
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm  # For progress bar visualization

plt.close('all')


# =============================================================================
# class
# =============================================================================

import os
import pandas as pd
import numpy as np

# Load Excel file
df = pd.read_excel('class.xlsx')

# Get patient codes - Read the "code" column
patient_codes = df['code'].dropna().unique()
patient_impaction = df['impaction'].dropna().unique()


# Check if images exist
existing_images = []  # List of existing images
missing_images = []   # List of missing images
image_folder = 'images'  # Folder containing images

image_name_class = []
for code in patient_codes:
    png_path = os.path.join(image_folder, f"{code}.png")
    jpg_path = os.path.join(image_folder, f"{code}.jpg")
    
    if os.path.exists(png_path) or os.path.exists(jpg_path):
        existing_images.append(str(code))  # If the image exists, add to the existing list
        if os.path.exists(png_path):
            image_name_class.append(f"{code}.png")
        if os.path.exists(jpg_path):
            image_name_class.append(f"{code}.jpg")
    else:
        missing_images.append(code)  # If the image does not exist, add to the missing list

# Calculate unique and duplicate counts for existing images
existing_unique_count = len(set(existing_images))  # Number of unique codes in existing images
existing_duplicate_count = len(existing_images) - existing_unique_count  # Number of duplicate codes

# Calculate unique and duplicate counts for missing images
missing_unique_count = len(set(missing_images))  # Number of unique codes in missing images
missing_duplicate_count = len(missing_images) - missing_unique_count  # Number of duplicate codes




# Print results

# print("Missing images:", missing_images)
print("\nStatistics:")
print(f"Existing images - Unique: {existing_unique_count}, Duplicates: {existing_duplicate_count}")
print(f"Missing images - Unique: {missing_unique_count}, Duplicates: {missing_duplicate_count}")




# existing_images=list (set(existing_images))
# missing_images=list (set(missing_images))



# =============================================================================
# Localization
# =============================================================================


# Configuration flags
enable_show = False  # Set to True to display processed images
enable_save = False  # Set to True to save output images


# enable_show = True   
# enable_save = True   



# Mask counters
dual_mask = 0  # Counts images with both L and R masks
single_mask = 0  # Counts images with only one mask

# Path configuration
folder_path = 'images'

# Get list of all image files in directory
image_name_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
all_entries = []

# Read annotation data from CSV file
data = []
with open('localize.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append({
            'image': row['image_name'],
            'x': int(row['bbox_x']),
            'y': int(row['bbox_y']),
            'w': int(row['bbox_width']),
            'h': int(row['bbox_height']),
            'label': row['label_name']
        })

# Group annotations by image name
image_data = defaultdict(list)
for entry in data:
    if entry['image'] in image_name_list:
        image_data[entry['image']].append(entry)

# Calculate total number of images for progress tracking
total_images = len(image_data)
processed_images = 0

print(f"Starting processing of {total_images} images...")

def create_mask(img_shape, x, y, w, h):
    """Create a binary mask for given coordinates"""
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x+w), min(height, y+h)
    mask[y1:y2, x1:x2] = 255
    return mask, (x1, y1, x2, y2)

# Process each image with progress tracking
for image_name, entries in tqdm(image_data.items(), desc="Processing images"):
    img_path = os.path.join(folder_path, image_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"\nError reading image: {img_path}")
        continue
    
    height, width = img.shape[:2]
    
    if enable_show:
        # Create figure with 2x2 subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Original Image
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image', fontsize=12)
        plt.axis('off')
        
        # Initialize output images
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        masked_img = img.copy()
        contour_img = img.copy()
        
        for entry in entries:
            x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']
            label = entry['label']
            
            # Set visualization parameters based on label
            if label == 'L':
                color = (0, 0, 255)  # Red for L
                thickness = 2
            elif label == 'R':
                color = (0, 255, 0)  # Green for R
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue for others
                thickness = 1
            
            # Create mask and get valid coordinates
            mask, (x1, y1, x2, y2) = create_mask(img.shape, x, y, w, h)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Draw rectangle on masked image
            cv2.rectangle(masked_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, color, thickness)
        
        # 2. Combined Mask
        plt.subplot(2, 2, 2)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Mask (All Regions)', fontsize=12)
        plt.axis('off')
        
        # 3. Masked Image
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Regions', fontsize=12)
        plt.axis('off')
        
        # 4. Contour Image
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Contours\n(Red: L, Green: R)', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        
        if enable_save:
            # Save results
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'result_{image_name}')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
    
    # Update counters
    all_entries.append(len(entries))
    
    if len(entries) == 2:
        dual_mask += 1
    elif len(entries) == 1:
        single_mask += 1
    
    # Update progress
    # processed_images += 1
    # progress = (processed_images / total_images) * 100
    
    # Print periodic updates
    # if processed_images % 40 == 0 or processed_images == total_images:
    #     print(f"\nProcessed: {processed_images}/{total_images} ({progress:.1f}%)")
    #     print(f"Images with single mask: {single_mask}")
    #     print(f"Images with dual masks: {dual_mask}")

# Print final summary
print("\nFinal Results:")
print(f"Total images processed: {total_images}")
print(f"Images with single mask: {single_mask} ({single_mask/total_images*100:.1f}%)")
print(f"Images with dual masks: {dual_mask} ({dual_mask/total_images*100:.1f}%)")


 
# =============================================================================
# Prepare YOLO Dataset (Non-Functional Version)
# =============================================================================

import shutil
from sklearn.model_selection import train_test_split

# Create directories for YOLO dataset
yolo_dir = 'yolo_dataset'
images_dir = os.path.join(yolo_dir, 'images')
labels_dir = os.path.join(yolo_dir, 'labels')


# Create directories if they don't exist
os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Create train/val/test directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Debug: Print some information to understand the issue
print("\nDebug Information:")
print(f"Total images in image_data: {len(image_data)}")
print(f"Total existing_images: {len(existing_images)}")

# Filter images with exactly 2 masks (L and R) that exist in our class data
valid_images = []
for img_name in image_data:
    if len(image_data[img_name]) == 2:  # Only images with both L and R
    
        code = os.path.splitext(img_name)[0]
        # print (code)
        if code in existing_images:
            valid_images.append(img_name)
            # print (code,' ++')

print(f"\nFound {len(valid_images)} images with dual masks and existing class information")

# If no valid images found, print more debug info and exit
if len(valid_images) == 0:
    print("\nError: No valid images found for YOLO dataset preparation. Possible reasons:")
    print("1. Mismatch between image names in localization data and class data")
    print("2. File extensions mismatch (.png vs .jpg)")
    print("3. No images have exactly two masks (L and R)")
    
    # Print sample of image names for comparison
    print("\nSample of image names in localization data:")
    print(list(image_data.keys())[:5])
    print("\nSample of existing image codes:")
    print(existing_images[:5])
    exit()

# Create a mapping from image code to impaction class
code_to_class = {}
for _, row in df.iterrows():
    code = row['code']
    # impaction = row['impaction']
    class_lable =row['class_lable']
    if pd.notna(code) and pd.notna(class_lable):
        # Convert impaction to class number
        if class_lable == 'A':
            class_num = 0
        elif class_lable == 'B':
            class_num = 1
        elif class_lable == 'C':
            class_num = 2
        elif class_lable == 'D':
            class_num = 3
        else:
            continue  # Skip invalid impaction values
        
        code_to_class[str(code)] = class_num

# Split dataset into train, val, test (70%, 15%, 15%)
try:
    train_val, test = train_test_split(valid_images, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # 0.1765*0.85=0.15
except ValueError as e:
    print(f"\nError during dataset split: {e}")
    print("This usually means there aren't enough valid images for splitting.")
    print("Possible solutions:")
    print("1. Check if your localization.csv and class.xlsx files are properly aligned")
    print("2. Verify that images with two masks exist in both datasets")
    print("3. Consider using all images for training if dataset is small")
    exit()

print(f"Dataset split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

# Process train set
print("\nProcessing train set...")
for img_name in tqdm(train):
    # Get image code (without extension)
    code = os.path.splitext(img_name)[0]
    
    # Get image path and read image
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Copy image to YOLO dataset
    dest_img_path = os.path.join(images_dir, 'train', img_name)
    shutil.copy(img_path, dest_img_path)
    
    # Prepare YOLO annotation file
    label_file = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(labels_dir, 'train', label_file)
    
    # Get impaction class for this image
    class_num = code_to_class.get(str (code), -1)
    if class_num == -1:
        print(f"No class found for image: {img_name}")
        continue
    
    # Get bounding boxes for this image
    entries = image_data[img_name]
    
    with open(label_path, 'w') as f:
        for entry in entries:
            x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']
            
            # Convert to YOLO format (normalized center coordinates and width/height)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Write to label file (class, x_center, y_center, width, height)
            f.write(f"{class_num} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

# Process val set
print("\nProcessing val set...")
for img_name in tqdm(val):
    # Get image code (without extension)
    code = os.path.splitext(img_name)[0]
    
    # Get image path and read image
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Copy image to YOLO dataset
    dest_img_path = os.path.join(images_dir, 'val', img_name)
    shutil.copy(img_path, dest_img_path)
    
    # Prepare YOLO annotation file
    label_file = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(labels_dir, 'val', label_file)
    
    # Get impaction class for this image
    class_num = code_to_class.get(code, -1)
    if class_num == -1:
        print(f"No class found for image: {img_name}")
        continue
    
    # Get bounding boxes for this image
    entries = image_data[img_name]
    
    with open(label_path, 'w') as f:
        for entry in entries:
            x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']
            
            # Convert to YOLO format (normalized center coordinates and width/height)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Write to label file (class, x_center, y_center, width, height)
            f.write(f"{class_num} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

# Process test set
print("\nProcessing test set...")
for img_name in tqdm(test):
    # Get image code (without extension)
    code = os.path.splitext(img_name)[0]
    
    # Get image path and read image
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Copy image to YOLO dataset
    dest_img_path = os.path.join(images_dir, 'test', img_name)
    shutil.copy(img_path, dest_img_path)
    
    # Prepare YOLO annotation file
    label_file = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(labels_dir, 'test', label_file)
    
    # Get impaction class for this image
    class_num = code_to_class.get(code, -1)
    if class_num == -1:
        print(f"No class found for image: {img_name}")
        continue
    
    # Get bounding boxes for this image
    entries = image_data[img_name]
    
    with open(label_path, 'w') as f:
        for entry in entries:
            x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']
            
            # Convert to YOLO format (normalized center coordinates and width/height)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Write to label file (class, x_center, y_center, width, height)
            f.write(f"{class_num} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

print("\nYOLO dataset preparation completed successfully!")
print(f"Dataset saved in: {yolo_dir}") 