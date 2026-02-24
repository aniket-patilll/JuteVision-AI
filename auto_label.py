import os
import cv2
from ultralytics import YOLO

# Paths
IMAGE_DIR = "backend/training_data/images"
LABEL_DIR = "backend/training_data/labels"
MODEL_PATH = "backend/models/sacks_custom.pt"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Load existing model to give us a head start
print(f"Loading model {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
print(f"Found {len(image_files)} images to label.")

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # Run prediction (low confidence to get as many boxes as possible to start with)
    results = model.predict(img_path, conf=0.10, iou=0.5)
    
    # Save YOLO format labels (class_id center_x center_y width height)
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    
    with open(label_path, "w") as f:
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywhn.cpu().numpy()  # Normalized xywh
            cls_ids = results[0].boxes.cls.cpu().numpy()
            
            for box, cls_id in zip(boxes, cls_ids):
                # Class ID 0 is sack
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
    
    print(f"Auto-labeled {img_name} -> {len(results[0].boxes)} boxes.")

print("\n--- AUTO-LABELING COMPLETE ---")
print(f"Labels saved to: {LABEL_DIR}")
print("Next step: Use LabelImg to correct any missed or incorrect boxes!")
