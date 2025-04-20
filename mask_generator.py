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
        existing_images.append(code)  # If the image exists, add to the existing list
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

print("Missing images:", missing_images)
print("\nStatistics:")
print(f"Existing images - Unique: {existing_unique_count}, Duplicates: {existing_duplicate_count}")
print(f"Missing images - Unique: {missing_unique_count}, Duplicates: {missing_duplicate_count}")




existing_images=list (set(existing_images))
missing_images=list (set(missing_images))



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
            output_dir = 'output_mask_generator'
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