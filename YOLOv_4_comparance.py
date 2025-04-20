import torch
from ultralytics import YOLO
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sklearn.metrics import jaccard_score
import pandas as pd
from PIL import Image
import time

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================
# 1. Configuration
# =============================================================================
models = ['yolov10n', 'yolov8n', 'yolov9c', ]
config = {
    'epochs': 20,
    'imgsz': 640,
    'batch': 8,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dataset_path': 'yolo_dataset',
    'test_images': ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg'],
    'output_dir': 'model_comparison'
}

# =============================================================================
# 2. Training Function for Multiple Models
# =============================================================================
def train_models():
    # Create dataset config
    dataset_path = os.path.abspath(config['dataset_path'])
    yolo_config = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'images/train'),
        'val': os.path.join(dataset_path, 'images/val'),
        'names': {0: 'A', 1: 'B', 2: 'C'},
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

    # Verify dataset structure
    required_dirs = [
        os.path.join(dataset_path, 'images/train'),
        os.path.join(dataset_path, 'images/val'),
        os.path.join(dataset_path, 'labels/train'),
        os.path.join(dataset_path, 'labels/val')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required dataset directory not found: {dir_path}")

    # Train each model
    results = {}
    for model_name in models:
        print(f"\n{'='*40}")
        print(f"Training {model_name} model")
        print(f"{'='*40}")
        
        # Create model-specific output directory
        model_dir = os.path.join(config['output_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'test_results'), exist_ok=True)
        
        # Initialize and train model
        start_time = time.time()
        model = YOLO(f'{model_name}.pt')
        
        train_results = model.train(
            data='dataset_config.yaml',
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            project=model_dir,
            name='train',
            exist_ok=True,
            val=True,
            plots=True
        )
        
        # Evaluate model
        metrics = model.val()
        training_time = time.time() - start_time
        
        # Store results
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'training_time': training_time,
            'test_results': []
        }
        
        # Save model info
        with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"mAP50: {metrics.box.map50:.4f}\n")
            f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        
    return results

# =============================================================================
# 3. Test Models on Sample Images
# =============================================================================
def test_models(trained_models):
    # Create test images directory if not exists
    test_images_dir = os.path.join(config['output_dir'], 'test_images')
    os.makedirs(test_images_dir, exist_ok=True)
    
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        model_dir = os.path.join(config['output_dir'], model_name)
        
        for img_name in config['test_images']:
            # Check if test image exists
            img_path = os.path.join('test_images', img_name)
            if not os.path.exists(img_path):
                print(f"Test image {img_name} not found, skipping...")
                continue
            
            # Run prediction
            results = model.predict(img_path, conf=0.5, save=True, save_txt=True, 
                                  project=os.path.join(model_dir, 'test_results'),
                                  name=os.path.splitext(img_name)[0])
            
            # Store results
            result = results[0]
            if result.boxes is not None:
                detected_objects = len(result.boxes)
                avg_conf = sum(result.boxes.conf) / detected_objects if detected_objects > 0 else 0
            else:
                detected_objects = 0
                avg_conf = 0
                
            model_data['test_results'].append({
                'image': img_name,
                'detected_objects': detected_objects,
                'average_confidence': avg_conf,
                'result_path': os.path.join(model_dir, 'test_results', os.path.splitext(img_name)[0], img_name)
            })
            
            # Save visualization with ground truth if available
            visualize_prediction(model, img_path, model_dir, img_name)
    
    return trained_models

# =============================================================================
# 4. Visualization Function
# =============================================================================
def visualize_prediction(model, img_path, output_dir, img_name):
    # Read image
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Check for ground truth
    label_path = img_path.replace('images', 'labels').replace(os.path.splitext(img_path)[1], '.txt')
    has_ground_truth = os.path.exists(label_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if has_ground_truth else 1, figsize=(15, 8))
    fig.suptitle(f"Model: {os.path.basename(output_dir)}\nImage: {img_name}", fontsize=14)
    
    if not has_ground_truth:
        axes = [axes]
    
    # Run prediction
    results = model.predict(img_path, conf=0.5)
    result = results[0]
    
    # Plot prediction
    pred_img = image.copy()
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(pred_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
    
    axes[0].imshow(pred_img)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    
    # Plot ground truth if available
    if has_ground_truth:
        gt_img = image.copy()
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert YOLO format to image coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(gt_img, model.names[int(cls_id)], (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[1].imshow(gt_img)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_results', f'comparison_{img_name}'))
    plt.close()

# =============================================================================
# 5. Create Comparison Report
# =============================================================================
def create_comparison_report(trained_models):
    # Create DataFrame for metrics comparison
    comparison_data = []
    
    for model_name, model_data in trained_models.items():
        metrics = model_data['metrics']
        
        # Get test results summary
        total_detected = sum(r['detected_objects'] for r in model_data['test_results'])
        avg_conf = sum(r['average_confidence'] for r in model_data['test_results']) / len(model_data['test_results'])
        
        comparison_data.append({
            'Model': model_name,
            'Training Time (s)': model_data['training_time'],
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Precision': metrics.box.p.mean(),
            'Recall': metrics.box.r.mean(),
            'Total Detected (6 images)': total_detected,
            'Average Confidence': avg_conf,
            'Model Size (MB)': os.path.getsize(f'{model_name}.pt') / (1024 * 1024)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to Excel
    excel_path = os.path.join(config['output_dir'], 'model_comparison.xlsx')
    df.to_excel(excel_path, index=False)
    
    # Print report
    print("\nModel Comparison Results:")
    print("="*80)
    print(df.to_markdown(tablefmt="grid", stralign="center", numalign="center"))
    print(f"\nFull comparison report saved to: {excel_path}")
    
    return df

# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    try:
        # 1. Train all models
        trained_models = train_models()
        
        # 2. Test models on sample images
        trained_models = test_models(trained_models)
        
        # 3. Create comparison report
        comparison_df = create_comparison_report(trained_models)
        
        print("\nAll operations completed successfully!")
        print(f"Results saved in: {os.path.abspath(config['output_dir'])}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please ensure:")
        print("1. Your dataset is properly organized in YOLO format")
        print("2. Test images exist in 'test_images' directory")
        print("3. You have internet connection to download models if needed")