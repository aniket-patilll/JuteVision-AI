# YOLOv8 Custom Training Guide - Jute Bag Detection

This guide will help you train a custom YOLOv8 model to accurately detect and count jute bags.

## 📋 Overview

Training a custom model involves 4 main steps:
1. **Collect Images** - Gather warehouse photos
2. **Annotate Images** - Label jute bags with bounding boxes
3. **Train Model** - Fine-tune YOLOv8 on your data
4. **Deploy Model** - Replace the default model in your app

---

## Step 1: Collect Training Images (100-200 images)

### What to Capture
- Different angles of your warehouse
- Various lighting conditions (morning, afternoon, evening)
- Different bag arrangements (stacked, pallets, scattered)
- Close-up and wide shots
- Include empty areas too (for negative examples)

### How to Collect
```bash
# Option 1: Extract frames from existing warehouse videos
cd /Users/psaipratyusha/Desktop/JuteVision_AI/backend
mkdir -p training_data/images

# Extract 1 frame every 2 seconds from a video
ffmpeg -i your_warehouse_video.mp4 -vf fps=0.5 training_data/images/frame_%04d.jpg
```

### Recommended Dataset Split
- **Training:** 80% (80-160 images)
- **Validation:** 20% (20-40 images)

---

## Step 2: Annotate Images

### Option A: Using Roboflow (Easiest - Recommended)

1. **Sign up at [Roboflow](https://roboflow.com/)** (Free tier available)

2. **Create a new project:**
   - Click "Create New Project"
   - Name: "Jute Bag Detection"
   - Project Type: "Object Detection"
   - Annotation Group: "jute_bag"

3. **Upload your images:**
   - Drag and drop all collected images
   - Wait for upload to complete

4. **Annotate (draw bounding boxes):**
   - Click on each image
   - Draw rectangles around each jute bag
   - Label as "jute_bag"
   - Save and move to next image
   - Keyboard shortcuts: `b` (box), `Enter` (save)

5. **Generate dataset:**
   - Click "Generate" → "Version 1"
   - Preprocessing: "Auto-Orient", "Resize to 640x640"
   - Augmentation (optional): "Flip: Horizontal", "Brightness: ±15%"
   - Click "Generate"
   - Export Format: "YOLOv8"
   - Download the dataset

### Option B: Using LabelImg (Local Tool)

```bash
# Install LabelImg
pip install labelImg

# Run LabelImg
labelImg training_data/images
```

**Steps:**
1. Click "Open Dir" → Select `training_data/images`
2. Click "Change Save Dir" → Select `training_data/labels`
3. Set format to "YOLO"
4. Click "Create RectBox" (or press `w`)
5. Draw boxes around jute bags
6. Label as "jute_bag"
7. Press `Ctrl+S` to save
8. Press `d` for next image

---

## Step 3: Train the Model

### Dataset Structure
Your dataset should look like this:
```
training_data/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

### Create Dataset Config File

Create `training_data/data.yaml`:
```yaml
# Dataset paths
path: /Users/psaipratyusha/Desktop/JuteVision_AI/backend/training_data
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['jute_bag']  # class names
```

### Training Script

Create `backend/train_model.py`:
```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')  # medium model (good balance)

# Train the model
results = model.train(
    data='training_data/data.yaml',  # path to dataset config
    epochs=50,                        # number of training epochs
    imgsz=640,                        # image size
    batch=16,                         # batch size (reduce if GPU memory issues)
    name='jute_bag_detector',         # experiment name
    device='mps',                     # use 'mps' for Mac, 'cuda' for NVIDIA GPU, 'cpu' for CPU
    patience=10,                      # early stopping patience
    save=True,                        # save checkpoints
    plots=True,                       # save training plots
    cache=True,                       # cache images for faster training
)

print("Training complete!")
print(f"Best model saved to: runs/detect/jute_bag_detector/weights/best.pt")
```

### Run Training

```bash
cd /Users/psaipratyusha/Desktop/JuteVision_AI/backend
python3 train_model.py
```

**Training Time Estimate:**
- Mac M1/M2 (MPS): 30-60 minutes for 50 epochs
- CPU only: 2-4 hours
- NVIDIA GPU: 15-30 minutes

### Monitor Training

Watch the terminal output for:
- `mAP50` - Should increase (target: >0.85)
- `mAP50-95` - Overall accuracy (target: >0.70)
- `precision` - Accuracy of detections (target: >0.85)
- `recall` - Percentage of bags found (target: >0.80)

Training plots saved to: `runs/detect/jute_bag_detector/`

---

## Step 4: Deploy Your Custom Model

### Test the Model First

```python
from ultralytics import YOLO

# Load your custom model
model = YOLO('runs/detect/jute_bag_detector/weights/best.pt')

# Test on a sample image
results = model('training_data/images/val/sample.jpg')

# Show results
results[0].show()
```

### Deploy to Your Application

1. **Copy the trained model:**
```bash
cp runs/detect/jute_bag_detector/weights/best.pt models/jute_bag_custom.pt
```

2. **Update `backend/app/tracker.py`:**
```python
# Line 10 - change the default model
def __init__(self, model_name="models/jute_bag_custom.pt"):  # Your custom model
```

3. **Restart your backend server:**
```bash
cd /Users/psaipratyusha/Desktop/JuteVision_AI/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test the new model** by uploading a warehouse video!

---

## 🎯 Tips for Better Results

### Image Collection
- ✅ Include variety: different times of day, angles, distances
- ✅ Capture edge cases: partially visible bags, shadows
- ❌ Avoid: blurry images, extreme angles

### Annotation
- ✅ Be consistent: always include the whole bag in the box
- ✅ Label all visible bags, even partially occluded ones
- ❌ Don't: draw boxes too tight or too loose

### Training
- Start with **yolov8m.pt** (medium) for good accuracy/speed balance
- If underfitting (low accuracy): Use **yolov8l.pt** or **yolov8x.pt**
- If overfitting: Add more augmentation, reduce epochs
- Monitor validation metrics, not just training loss

### Common Issues

**Problem: Low mAP scores (<0.5)**
- Solution: Add more diverse training images
- Solution: Check annotation quality
- Solution: Increase epochs to 100

**Problem: GPU/Memory errors**
- Solution: Reduce batch size to 8 or 4
- Solution: Use smaller model (yolov8s.pt)

**Problem: Model detects everything as jute bags**
- Solution: Add negative examples (images without bags)
- Solution: Increase confidence threshold during inference

---

## 📊 Expected Results

With a well-trained custom model:
- **Precision:** 85-95%
- **Recall:** 80-90%
- **Count Accuracy:** ±5% of actual count
- **Processing Speed:** 20-30 FPS on Mac M1/M2

---

## 🔄 Iterative Improvement

1. **Deploy model** → Test on real videos
2. **Identify errors** → Bags missed or false positives
3. **Add hard examples** → Collect more images of missed cases
4. **Re-train** → Include new images in dataset
5. **Repeat** until satisfied with accuracy

---

## 📚 Additional Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/modes/train/)
- [Roboflow Annotation Guide](https://roboflow.com/annotate)
- [YOLOv8 Training Tips](https://github.com/ultralytics/ultralytics/wiki/Tips-for-Best-Training-Results)

---

## 🆘 Need Help?

If you encounter issues during training:
1. Check the training logs in `runs/detect/jute_bag_detector/`
2. Verify your data.yaml file paths are correct
3. Ensure annotations match image filenames
4. Try a smaller batch size if memory errors occur

Good luck with your training! 🚀
