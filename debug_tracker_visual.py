
import cv2
from ultralytics import YOLO

# Path to the specific uploaded video
video_path = "/Users/psaipratyusha/Desktop/JuteVision_AI/backend/temp_uploads/4b491ffe-4800-407a-b89b-26e6d7ea9367_1103446707-preview.mp4"
model_path = "backend/models/sacks_custom.pt"
output_image = "debug_detection.jpg"

print(f"Loading model from {model_path}...")
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
max_detections = -1
best_frame = None

print("Scanning video for frame with most detections...")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference on every 5th frame to save time
    if frame_idx % 5 == 0:
        results = model.track(frame, persist=True, conf=0.5, iou=0.5, tracker="bytetrack.yaml", verbose=False)
        
        if results and results[0].boxes:
            count = len(results[0].boxes)
            if count > max_detections:
                max_detections = count
                # Plot matches the tracker's visualization
                best_frame = results[0].plot()
                print(f"Frame {frame_idx}: Found {count} bags (New Max)")

    frame_idx += 1
    if frame_idx > 200: # Limit to first 200 frames for speed
        break

cap.release()

if best_frame is not None:
    cv2.imwrite(output_image, best_frame)
    print(f"Saved annotated frame with {max_detections} detections to {output_image}")
else:
    print("No detections found.")
