"""
Train YOLOv8 on Roboflow Sacks Dataset

This script fine-tunes YOLOv8 to detect sacks/jute bags using the downloaded Roboflow dataset.
"""

from ultralytics import YOLO
import os

# Check if dataset exists
dataset_path = 'training_data/Sacks.v2i.yolov8/data.yaml'
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    print("Please ensure the Sacks dataset is in training_data/Sacks.v2i.yolov8/")
    exit(1)

print("=" * 60)
print("Starting YOLOv8 Training for Sack Detection")
print("=" * 60)

# Load pre-trained YOLOv8 nano model (Much faster)
print("\n1. Loading pre-trained YOLOv8n model...")
model = YOLO('models/yolov8n.pt')

# Train the model
print("\n2. Starting FAST training...")
print("   - Dataset: Roboflow Sacks v2i")
print("   - Epochs: 30")
print("   - Image size: 416x416")
print("   - Device: MPS (Mac GPU acceleration)")
print("\nThis should take 5-10 minutes. 🚀\n")

results = model.train(
    data=dataset_path,               # Path to data.yaml
    epochs=30,                        # Reduced epochs
    imgsz=416,                        # Reduced image size
    batch=32,                         # Increased batch size for nano
    name='sacks_detector_fast',       # Experiment name
    device='cpu',                     # Use CPU (MPS has bugs on Mac with Nano model)
    amp=False,                        # Disable mixed precision for stability
    patience=10,                      # Early stopping patience
    save=True,                        # Save checkpoints
    plots=True,                       # Generate training plots
    cache=True,                       # Cache images for faster training
    verbose=True,                     # Show detailed progress
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)

# Show results
best_model_path = 'runs/detect/sacks_detector_fast/weights/best.pt'
print(f"\n✅ Best model saved to: {best_model_path}")
print(f"✅ Training plots saved to: runs/detect/sacks_detector_fast/")

# Validate the model
print("\n3. Running validation...")
metrics = model.val()

print("\n📊 Model Performance:")
print(f"   - mAP@50: {metrics.box.map50:.3f}")
print(f"   - mAP@50-95: {metrics.box.map:.3f}")
print(f"   - Precision: {metrics.box.mp:.3f}")
print(f"   - Recall: {metrics.box.mr:.3f}")

# Copy to models folder for deployment
print("\n4. Deploying model...")
import shutil
shutil.copy(best_model_path, 'models/sacks_custom.pt')
print("✅ Model copied to: models/sacks_custom.pt")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Update backend/app/tracker.py line 10:")
print('   def __init__(self, model_name="models/sacks_custom.pt"):')
print("\n2. Restart your backend server")
print("\n3. Upload a warehouse video and test!")
print("\n🎉 You're ready to count jute bags accurately!")
