import cv2
import torch
from ultralytics import YOLO
try:
    from backend.app.utils import get_centroid, annotate_frame
except ImportError:
    from .utils import get_centroid, annotate_frame

class JuteBagTracker:
    def __init__(self, model_name="backend/models/sacks_custom.pt"):  # Custom sacks model
        print("Initializing JuteBagTracker (Custom Sacks Model)...")
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(model_name)
            print(f"YOLOv8 loaded successfully from {model_name}")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self.model = None

        # Persistent Counting State
        self.counted_ids = set()
        self.total_count = 0
        
        # Track history
        self.track_history = {}

    def _get_device(self):
        """Dynamic Device Setup: Mac (MPS), CUDA, or CPU."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def process_video(self, video_path, output_path, line_y=500, mode="static", on_update=None):
        """
        Process video using YOLOv8 tracking.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated output
            line_y: Y-coordinate of counting line (for conveyor mode)
            mode: "static" (count all unique bags) or "conveyor" (count line crossings)
            on_update: Callback function for real-time updates
            
        Returns:
            dict: Processing results including final count
        """
        if not self.model:
            print("Model not loaded.")
            return {"count": 0, "status": "failed", "error": "Model not loaded"}

        print(f"Processing video: {video_path} (Mode: {mode})")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Output saver
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        detection_count = 0
        
        # Local Counting State (Reset per video)
        current_count = 0
        counted_ids = set()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Run YOLOv8 tracking with OPTIMIZED parameters
            # conf=0.6: Higher confidence to ignore partial/weak boxes
            # iou=0.5: Standard NMS threshold
            # agnostic_nms=True: Prevents multiple boxes on same object even if class differs (though we only have 1 class)
            # classes=[0]: Ensure we only track "sack" class
            results = self.model.track(frame, persist=True, conf=0.6, iou=0.5, 
                                     tracker="bytetrack.yaml", 
                                     agnostic_nms=True,
                                     classes=[0],
                                     verbose=False)
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                detection_count += 1
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                confs = results[0].boxes.conf.cpu().tolist()
                
                # DEBUG: Print detections for tuning
                if frame_idx % 30 == 0:
                     print(f"[DEBUG] Frame {frame_idx}: Found {len(boxes)} boxes. Confs: {[f'{c:.2f}' for c in confs]}")
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                
                # Visualize results on the frame
                annotated_frame = results[0].plot()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    cx, cy = float(x), float(y)
                    
                    # Draw centroid
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    
                    # MODE-SPECIFIC COUNTING LOGIC
                    if mode == "static":
                        # Static Mode: Count any new unique bag ID
                        if track_id not in counted_ids:
                            current_count += 1
                            counted_ids.add(track_id)
                            print(f"Frame {frame_idx}: New bag {track_id} detected! Total: {current_count}")
                            
                            if on_update:
                                try:
                                    on_update({"count": current_count, "frame_idx": frame_idx})
                                except Exception as e:
                                    print(f"Callback error: {e}")
                    
                    elif mode == "conveyor":
                        # Conveyor Mode: Count bags crossing the line
                        if cy > line_y and track_id not in counted_ids:
                            current_count += 1
                            counted_ids.add(track_id)
                            print(f"Frame {frame_idx}: Bag {track_id} crossed line! Total: {current_count}")
                            
                            if on_update:
                                try:
                                    on_update({"count": current_count, "frame_idx": frame_idx})
                                except Exception as e:
                                    print(f"Callback error: {e}")
                
                # Replace frame with annotated one
                frame = annotated_frame

            # Draw counting info (show line only in conveyor mode)
            if mode == "conveyor":
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
                cv2.putText(frame, f"Conveyor Count: {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"Total Bags: {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        
        # Update global total for stream/persistence if needed, but return local for this video
        self.total_count += current_count 
        
        print(f"Processed video saved to {output_path} | Find Count: {current_count} | Frames with detections: {detection_count}/{frame_idx}")
        return {"count": current_count, "status": "completed"}

    # Generator for future streaming support
    # def process_video_generator(self, video_path, line_y=500):
    #     ... implementation deferred ...
    #     pass
