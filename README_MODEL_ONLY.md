# Packaging Quality Assurance Model - User Guide

> AI-Driven Packaging Quality Assurance System using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Information](#model-information)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Input Requirements](#input-requirements)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This AI-powered model automatically detects and classifies packaging types, safety labels, and damage in export packages. Designed specifically for Sri Lankan SMEs to reduce export rejections and ensure international compliance.

### Detected Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | `carton_box` | Standard corrugated cardboard boxes |
| 1 | `shrink_wrapped_pallet` | Pallet loads wrapped in shrink film |
| 2 | `pallet_stack` | Stacked pallets with goods |
| 3 | `fragile_label` | Fragile handling warning labels |
| 4 | `this_side_up_label` | Orientation indicators |
| 5 | `keep_dry_label` | Moisture protection warnings |
| 6 | `handle_with_care_label` | General care instructions |
| 7 | `package_damage` | Visible damage or defects |

---

## 📊 Model Information

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 0.73 | Mean Average Precision at IoU=0.5 |
| **mAP50-95** | 0.60 | Mean Average Precision at IoU=0.5:0.95 |
| **Precision** | 0.95 | 95% of detections are correct |
| **Recall** | 0.65 | Detects 65% of all objects |
| **Inference Speed** | ~50ms | Per 640x640 image on GPU |
| **Model Size** | 22 MB | YOLOv8s architecture |

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- CPU (Intel i5 or equivalent)

**Recommended:**
- Python 3.9+
- 8GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.0+ and cuDNN

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
pip install ultralytics opencv-python numpy pillow
```

### Step 2: Download Model File

Place the `best.pt` model file (22 MB) in your working directory.

### Step 3: Verify Installation

```python
from ultralytics import YOLO

# Test model loading
model = YOLO('best.pt')
print("✅ Model loaded successfully!")
```

---

## 🚀 Basic Usage

### Single Image Prediction

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference
results = model.predict('package_image.jpg', conf=0.684)

# Process results
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        
        print(f"Detected: {class_name}")
        print(f"Confidence: {confidence:.2f}")
```

**Output:**
```
Detected: carton_box
Confidence: 0.92
Detected: fragile_label
Confidence: 0.87
Detected: keep_dry_label
Confidence: 0.79
```

---

## 📥 Input Requirements

### Image Specifications

| Property | Requirement | Optimal |
|----------|-------------|---------|
| **Format** | JPG, PNG, BMP, TIFF | JPG |
| **Size** | Min 320x320px | 640x640px |
| **Color Space** | RGB | RGB |
| **File Size** | Max 10MB | 1-3MB |
| **Quality** | Good lighting, in-focus | Well-lit, sharp |

### Best Practices

✅ **Do:**
- Use good lighting (avoid dark/shadowy images)
- Capture full package in frame
- Keep camera steady (avoid blur)
- Include all labels clearly
- Use 640x640 or higher resolution

❌ **Don't:**
- Use extremely low resolution (<320px)
- Upload heavily compressed images
- Use motion-blurred images
- Crop out important parts
- Use extreme angles

---

## 📤 Output Format

### Results Object Structure

```python
results = model.predict('image.jpg')

# Access results
for result in results:
    # Boxes information
    boxes = result.boxes          # All detected boxes
    names = result.names           # Class names dictionary
    orig_img = result.orig_img     # Original image
    
    # For each box
    for box in boxes:
        # Coordinates
        xyxy = box.xyxy[0]         # [x1, y1, x2, y2]
        xywh = box.xywh[0]         # [x_center, y_center, width, height]
        
        # Classification
        cls = int(box.cls[0])      # Class ID (0-7)
        conf = float(box.conf[0])  # Confidence (0-1)
        
        # Get class name
        class_name = result.names[cls]
```

### Example Output

```python
{
    "class_id": 0,
    "class_name": "carton_box",
    "confidence": 0.92,
    "bbox": {
        "x1": 123.4,
        "y1": 56.7,
        "x2": 456.8,
        "y2": 789.0
    }
}
```

---

## ⚙️ Configuration

### Parameters

```python
results = model.predict(
    source='image.jpg',      # Image path or numpy array
    conf=0.684,              # Confidence threshold (0-1)
    iou=0.45,                # IoU threshold for NMS
    imgsz=640,               # Image size for inference
    device='cpu',            # 'cpu' or 'cuda' or 0,1,2,3 (GPU ID)
    verbose=True,            # Print results
    save=False,              # Save annotated images
    save_txt=False,          # Save results as .txt
    project='runs/predict',  # Save directory
    name='exp'               # Experiment name
)
```

### Confidence Threshold Guide

| Threshold | Use Case | Behavior |
|-----------|----------|----------|
| **0.40** | High Recall | Catch more objects, more false positives |
| **0.684** | Balanced (Recommended) | Best F1-score, optimal precision-recall |
| **0.80** | High Precision | Fewer false positives, may miss some objects |

**Recommendation:** Use `conf=0.684` for best overall performance.

---

## 💡 Examples

### Example 1: Simple Detection

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Detect objects
results = model.predict('package.jpg', conf=0.684)

# Print all detections
for r in results:
    print(f"Found {len(r.boxes)} objects:")
    for box in r.boxes:
        name = r.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  - {name}: {conf*100:.1f}%")
```

**Output:**
```
Found 3 objects:
  - carton_box: 92.3%
  - fragile_label: 87.5%
  - keep_dry_label: 79.2%
```

---

### Example 2: Get Bounding Box Coordinates

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('package.jpg', conf=0.684)

for result in results:
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Get class info
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        
        print(f"\n{class_name}:")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Top-left: ({x1:.0f}, {y1:.0f})")
        print(f"  Bottom-right: ({x2:.0f}, {y2:.0f})")
```

**Output:**
```
carton_box:
  Confidence: 0.92
  Top-left: (123, 57)
  Bottom-right: (457, 789)

fragile_label:
  Confidence: 0.87
  Top-left: (234, 123)
  Bottom-right: (346, 235)
```

---

### Example 3: Check for Damage

```python
from ultralytics import YOLO

def check_package_quality(image_path):
    """
    Check if package has damage
    Returns: (has_damage, all_detections)
    """
    model = YOLO('best.pt')
    results = model.predict(image_path, conf=0.684)
    
    detections = []
    has_damage = False
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Check for damage (class_id = 7)
            if class_id == 7:
                has_damage = True
            
            detections.append({
                'class': class_name,
                'confidence': confidence
            })
    
    return has_damage, detections

# Use function
has_damage, detections = check_package_quality('package.jpg')

if has_damage:
    print("⚠️  WARNING: Package damage detected!")
else:
    print("✅ Package appears undamaged")

print(f"\nTotal detections: {len(detections)}")
for det in detections:
    print(f"  - {det['class']}: {det['confidence']:.2f}")
```

**Output:**
```
⚠️  WARNING: Package damage detected!

Total detections: 3
  - carton_box: 0.92
  - fragile_label: 0.87
  - package_damage: 0.85
```

---

### Example 4: Batch Processing Multiple Images

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('best.pt')

# Get all images in folder
image_folder = Path('images/')
image_files = list(image_folder.glob('*.jpg'))

print(f"Processing {len(image_files)} images...\n")

# Process all images
results = model.predict(
    source=str(image_folder),
    conf=0.684,
    save=True,  # Save annotated images
    project='results',
    name='batch_detection'
)

# Summarize results
total_detections = {}
for result in results:
    for box in result.boxes:
        class_name = result.names[int(box.cls)]
        total_detections[class_name] = total_detections.get(class_name, 0) + 1

print("Summary:")
print(f"  Total images: {len(image_files)}")
print(f"  Total objects: {sum(total_detections.values())}")
print("\nDetections by class:")
for class_name, count in sorted(total_detections.items()):
    print(f"  {class_name}: {count}")
```

**Output:**
```
Processing 25 images...

Summary:
  Total images: 25
  Total objects: 78

Detections by class:
  carton_box: 45
  fragile_label: 12
  handle_with_care_label: 8
  keep_dry_label: 7
  package_damage: 3
  shrink_wrapped_pallet: 3
```

---

### Example 5: Draw Detections on Image

```python
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# Run detection
results = model.predict('package.jpg', conf=0.684)

# Get annotated image
annotated_img = results[0].plot()  # Draws boxes and labels

# Save result
cv2.imwrite('result_annotated.jpg', annotated_img)

# Or display
cv2.imshow('Detections', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Saved annotated image to 'result_annotated.jpg'")
```

---

### Example 6: Use with NumPy Array

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

# Load image as numpy array
image = cv2.imread('package.jpg')

# Predict directly on array
results = model.predict(image, conf=0.684)

# Process results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
```

---

### Example 7: Video Processing

```python
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

# Open video
cap = cv2.VideoCapture('packaging_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on frame
    results = model.predict(frame, conf=0.684, verbose=False)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('Packaging QA', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Example 8: Command Line Usage

```bash
# Single image
yolo predict model=best.pt source=package.jpg conf=0.684

# Folder of images
yolo predict model=best.pt source=images/ conf=0.684 save=True

# Video
yolo predict model=best.pt source=video.mp4 conf=0.684 save=True

# Webcam
yolo predict model=best.pt source=0 conf=0.684 show=True
```

---

## 🔧 Troubleshooting

### Issue 1: "Model file not found"

**Problem:** 
```python
FileNotFoundError: [Errno 2] No such file or directory: 'best.pt'
```

**Solution:**
```python
# Use absolute path
from pathlib import Path

model_path = Path(__file__).parent / 'best.pt'
model = YOLO(str(model_path))

# Or check if file exists
import os
if not os.path.exists('best.pt'):
    print("❌ Model file not found!")
else:
    model = YOLO('best.pt')
```

---

### Issue 2: "CUDA out of memory"

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Solution 1: Use CPU
results = model.predict('image.jpg', device='cpu')

# Solution 2: Reduce batch size
results = model.predict(images, batch=1)

# Solution 3: Reduce image size
results = model.predict('image.jpg', imgsz=320)
```

---

### Issue 3: Low Detection Rate (Missing Objects)

**Problem:** Model not detecting all objects

**Solutions:**

```python
# Solution 1: Lower confidence threshold
results = model.predict('image.jpg', conf=0.40)  # Lower from 0.684

# Solution 2: Check image quality
import cv2
img = cv2.imread('image.jpg')
print(f"Image shape: {img.shape}")
print(f"Brightness: {img.mean()}")  # Should be 50-200

# Solution 3: Resize image if too small
if img.shape[0] < 640 or img.shape[1] < 640:
    img = cv2.resize(img, (640, 640))
    results = model.predict(img, conf=0.684)
```

---

### Issue 4: Too Many False Positives

**Problem:** Model detecting objects that aren't there

**Solutions:**

```python
# Solution 1: Increase confidence threshold
results = model.predict('image.jpg', conf=0.80)  # Higher from 0.684

# Solution 2: Filter by class
results = model.predict('image.jpg', conf=0.684)
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        # Only accept specific classes
        if class_id in [0, 3, 7]:  # carton_box, fragile_label, damage
            print(f"Valid detection: {r.names[class_id]}")
```

---

### Issue 5: Slow Inference

**Problem:** Predictions taking too long

**Solutions:**

```python
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Solution 1: Use GPU (automatically used if available)
model = YOLO('best.pt')  # Will use GPU if available

# Solution 2: Reduce image size
results = model.predict('image.jpg', imgsz=320)  # Faster

# Solution 3: Disable verbose output
results = model.predict('image.jpg', verbose=False)

# Solution 4: Batch processing (for multiple images)
results = model.predict(['img1.jpg', 'img2.jpg'], batch=16)
```

---

### Issue 6: Import Errors

**Problem:**
```python
ImportError: No module named 'ultralytics'
```

**Solution:**
```bash
# Install required packages
pip install ultralytics opencv-python numpy pillow

# Verify installation
python -c "from ultralytics import YOLO; print('✅ Installation successful')"
```

---

## 📊 Performance Optimization

### Speed vs Accuracy Trade-offs

```python
# Fastest (less accurate)
results = model.predict('image.jpg', imgsz=320, conf=0.5, device='cuda')

# Balanced (recommended)
results = model.predict('image.jpg', imgsz=640, conf=0.684, device='cuda')

# Most Accurate (slower)
results = model.predict('image.jpg', imgsz=1280, conf=0.40, device='cuda')
```

### Batch Processing for Speed

```python
from pathlib import Path

# Instead of processing one by one (slow)
for img in images:
    results = model.predict(img)  # ❌ Slow

# Process in batches (fast)
results = model.predict(images, batch=16)  # ✅ Fast
```

---

## 📖 Additional Information

### Model Architecture
- **Base Model:** YOLOv8s (Small)
- **Framework:** Ultralytics YOLOv8
- **Input Size:** 640x640 pixels
- **Output:** Bounding boxes + Class predictions

### Training Details
- **Dataset Size:** 106 images
- **Training Split:** 70% train, 15% val, 15% test
- **Epochs:** 150
- **Augmentation:** Extensive (rotation, scaling, color jitter)
- **Optimizer:** AdamW

### Classes Performance

| Class | Samples | Detection Rate |
|-------|---------|----------------|
| carton_box | 93 | ⭐⭐⭐⭐⭐ Excellent |
| package_damage | 17 | ⭐⭐⭐⭐ Good |
| pallet_stack | 24 | ⭐⭐⭐⭐ Good |
| keep_dry_label | 19 | ⭐⭐⭐ Fair |
| this_side_up_label | 18 | ⭐⭐⭐ Fair |
| handle_with_care_label | 16 | ⭐⭐⭐ Fair |
| fragile_label | 13 | ⭐⭐⭐ Fair |
| shrink_wrapped_pallet | 13 | ⭐⭐ Challenging |

---

## 👥 Authors

**Research Team:**
- S.J.G.I. Umeshika (EC/2021/017)
- D.T.D. Silva (EC/2021/058)
- K.D.T Jayathilaka (EC/2021/078)

**Supervisor:** Mr. Sukhitha Sandunwala

**Institution:** University of Kelaniya, Sri Lanka

---

## 📞 Support

For questions or issues:
- 📧 Email: support@packaging-qa.lk
- 🐛 Report Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

**Model Version:** v1.0.0  
**Last Updated:** February 10, 2026
