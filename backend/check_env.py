import os
import torch
import cv2
import numpy as np
import torchvision
try:
    from ultralytics import YOLO
    import ultralytics
except ImportError:
    YOLO = None
    ultralytics = None

def check_env():
    print("=== JuteVision AI Diagnostic Script ===")
    
    # 1. Check Libraries
    print(f"\n[1] Library Versions:")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Torchvision: {torchvision.__version__}")
    print(f"  - OpenCV: {cv2.__version__}")
    if ultralytics:
        print(f"  - Ultralytics: {ultralytics.__version__}")
    else:
        print("  - Ultralytics: NOT INSTALLED")
        
    # 2. Check Hardware Acceleration
    print(f"\n[2] Hardware Acceleration:")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  - MPS Available (Mac): {torch.backends.mps.is_available()}")
    
    # 3. Check Models
    print(f"\n[3] Model Verification:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    expected_models = ["sacks_custom.pt", "yolov8n.pt", "yolov8m.pt"]
    
    if not os.path.exists(models_dir):
        print(f"  - WARNING: 'models/' directory NOT FOUND at {models_dir}")
    else:
        for model in expected_models:
            model_path = os.path.join(models_dir, model)
            exists = os.path.exists(model_path)
            size = os.path.getsize(model_path) / (1024*1024) if exists else 0
            status = f"EXISTS ({size:.2f} MB)" if exists else "MISSING"
            print(f"  - {model}: {status}")
            
    # 4. Check Tracking Dependencies
    print(f"\n[4] Tracking Dependencies:")
    try:
        import lap
        print(f"  - lap library: INSTALLED (v{getattr(lap, '__version__', 'unknown')})")
    except ImportError:
        print("  - lap library: MISSING (Required for ByteTrack)")
        
    print("\n" + "="*40)
    print("Please share this output with the development team for comparison.")

if __name__ == "__main__":
    check_env()
