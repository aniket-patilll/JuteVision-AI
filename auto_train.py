import os
import shutil
import yaml
from ultralytics import YOLO

# Paths
BASE_DIR = "backend/training_data"
SRC_IMAGES = os.path.join(BASE_DIR, "images")
SRC_LABELS = os.path.join(BASE_DIR, "labels")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMG = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL = os.path.join(DATASET_DIR, "val", "labels")

# 1. Setup Dataset Structure
print("Setting up YOLO dataset structure...")
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# Copy all images and labels to both train and val (for overfitting demo purposes)
for filename in os.listdir(SRC_IMAGES):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        img_src = os.path.join(SRC_IMAGES, filename)
        lbl_name = os.path.splitext(filename)[0] + ".txt"
        lbl_src = os.path.join(SRC_LABELS, lbl_name)
        
        # Only copy if both image and label exist (even if label is empty, we touch it in auto_label)
        if os.path.exists(lbl_src):
            shutil.copy(img_src, os.path.join(TRAIN_IMG, filename))
            shutil.copy(img_src, os.path.join(VAL_IMG, filename))
            
            shutil.copy(lbl_src, os.path.join(TRAIN_LBL, lbl_name))
            shutil.copy(lbl_src, os.path.join(VAL_LBL, lbl_name))

# 2. Create data.yaml
yaml_path = os.path.abspath(os.path.join(DATASET_DIR, "data.yaml"))
data_yaml = {
    'path': os.path.abspath(DATASET_DIR),
    'train': 'train/images',
    'val': 'val/images',
    'nc': 1,
    'names': ['sack']
}

with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"data.yaml created at {yaml_path}")

# 3. Train Model
MODEL_PATH = "backend/models/sacks_custom.pt"
print(f"Loading existing model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print("Starting Fine-tuning (25 epochs)...")
results = model.train(
    data=yaml_path,
    epochs=25,
    imgsz=640,
    batch=4,
    project="backend/training_data",
    name="fine_tune",
    exist_ok=True
)

# 4. Copy newly trained model back to main location
# YOLOv8 default is to put projects inside 'runs/detect' if not run with absolute paths
NEW_MODEL = "runs/detect/backend/training_data/fine_tune/weights/best.pt"
print(f"Training complete. Updating {MODEL_PATH} with {NEW_MODEL}...")

# Create backup of old model just in case
shutil.copy(MODEL_PATH, "backend/models/sacks_custom_backup.pt")
# Overwrite with new model
shutil.copy(NEW_MODEL, MODEL_PATH)

print("✅ Model successfully trained and updated! The backend will Auto-Reload.")
