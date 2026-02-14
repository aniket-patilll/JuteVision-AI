import cv2
import torch
import numpy as np
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

    def reset_state(self):
        """Resets the tracker state to zero."""
        print("Resetting JuteBagTracker state...")
        self.counted_ids = set()
        self.total_count = 0
        self.track_history = {}
        return {"status": "reset", "count": 0}

    def _get_device(self):
        """Dynamic Device Setup: Mac (MPS), CUDA, or CPU."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def detect_with_tiling(self, frame):
        """
        Performs inference using tiling (SAHI-lite) to detect small objects.
        Splits frame into overlapping tiles + full frame, then merges results with NMS.
        """
        height, width = frame.shape[:2]
        
        # Define Overlapping Tiles (ensure objects on seams are detected)
        # We use a 4-tile grid with overlap + 1 center tile + 1 full frame
        
        # Split points with overlap
        x_mid_left = int(width * 0.45)
        x_mid_right = int(width * 0.55)
        y_mid_top = int(height * 0.45)
        y_mid_bottom = int(height * 0.55)
        
        tiles = [
            # Top-Left (extends past mid)
            (0, 0, x_mid_right, y_mid_bottom),
            # Top-Right (starts before mid)
            (x_mid_left, 0, width, y_mid_bottom),
            # Bottom-Left
            (0, y_mid_top, x_mid_right, height),
            # Bottom-Right
            (x_mid_left, y_mid_top, width, height),
            # Center Tile (focus on the middle pile)
            (int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)),
            # Full Frame (Context)
            (0, 0, width, height)
        ]
        
        all_boxes = []
        all_confs = []
        all_cls = []

        for tx1, ty1, tx2, ty2 in tiles:
            # Crop tile
            tile_img = frame[ty1:ty2, tx1:tx2]
            if tile_img.size == 0: continue
            
            # --- PREPROCESSING (Enhance Contrast) ---
            # Jute bags are often white-on-white. CLAHE helps separate them.
            try:
                lab = cv2.cvtColor(tile_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                tile_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            except Exception:
                pass # Fallback to original if enhancement fails
            
            # Run Inference
            # Conf 0.25: Lowered back to catch all bags. We rely on strict Edge/Shape filters to stop noise.
            results = self.model.predict(tile_img, conf=0.25, iou=0.45, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy() # Use xyxy for easy offsetting
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                
                # Offset coordinates back to full frame
                boxes[:, [0, 2]] += tx1
                boxes[:, [1, 3]] += ty1
                
                all_boxes.append(boxes)
                all_confs.append(confs)
                all_cls.append(clss)
        
        if not all_boxes:
            return torch.empty((0, 4)), []
            
        # Concatenate all detections
        all_boxes = torch.tensor(np.concatenate(all_boxes))
        all_confs = torch.tensor(np.concatenate(all_confs))
        all_cls = torch.tensor(np.concatenate(all_cls))
        
        # Apply NMS (Non-Maximum Suppression)
        keep_indices = torch.ops.torchvision.nms(all_boxes, all_confs, 0.45)
        
        final_boxes = all_boxes[keep_indices]
        
        # --- GEOMETRIC & POSITION FILTERING (Remove Walls/Noise) ---
        valid_boxes = []
        frame_area = width * height
        
        for box in final_boxes:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            area = w * h
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if w <= 0 or h <= 0: continue
            
            aspect_ratio = w / h
            
            # Criteria (SMART):
            # 1. Size: Reject huge (wall) or tiny (speck) objects.
            is_large = area > frame_area * 0.15 # Slightly looser than before
            is_tiny = area < frame_area * 0.001
            
            # 2. Shape: Bags are generally "squarish" (0.5 to 2.5). 
            bad_ar = (aspect_ratio < 0.5) or (aspect_ratio > 2.5)
            
            # 3. Position: Reject detections in the top 15% (Ceiling)
            is_high = cy < (height * 0.15)
            
            # 4. EDGE EXCLUSION: Walls usually touch the edges. Bags are in the middle.
            # Reject if box touches Left, Right, or Top edge (within 5 pixels)
            touches_edge = (x1 < 5) or (y1 < 5) or (x2 > width - 5)
            
            # 5. VERTICAL WALL FILTER: Reject objects that are "Tall" (height > 20% of screen)
            is_tall = h > (height * 0.20)

            if not (is_large or is_tiny or bad_ar or is_high or touches_edge or is_tall):
                valid_boxes.append(box)
                
        if len(valid_boxes) > 0:
            return torch.stack(valid_boxes), list(range(len(valid_boxes)))
        else:
            return torch.empty((0, 4)), []

            return torch.empty((0, 4)), []

    def process_live_frame(self, frame):
        """
        Processes a single frame from the live webcam feed.
        Uses SCANNING MODE (Blue Zone) to count bags entering the area.
        Updates global state directly.
        """
        if self.model is None:
            return frame

        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # --- SCANNING MODE (Center Zone Logic) ---
        # 1. Define Zone (Blue Box)
        zone_x1 = int(width * 0.2)
        zone_x2 = int(width * 0.8)
        zone_y1 = int(height * 0.1)
        zone_y2 = int(height * 0.9)
        
        cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
        cv2.putText(annotated_frame, "LIVE SCANNING ZONE", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 2. Run Tracking
        results = self.model.track(frame, persist=True, conf=0.6, iou=0.5, 
                                 tracker="bytetrack.yaml", 
                                 agnostic_nms=True,
                                 classes=[0],
                                 verbose=False)
        
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            # Use plot() for the base tracking visual (IDs, boxes)
            # We overlay on top of this
            base_plot = results[0].plot()
            # Blend or just copy the plot? Let's use plot() as base but we drew the zone on 'annotated_frame'.
            # simpler: Let's draw the zone on the plot result
            annotated_frame = base_plot
            cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "LIVE SCANNING ZONE", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                cx, cy = float(x), float(y)
                
                # Check Zone
                is_in_zone = (zone_x1 < cx < zone_x2) and (zone_y1 < cy < zone_y2)
                
                if is_in_zone:
                    if track_id not in self.counted_ids:
                        # NEW BAG
                        self.total_count += 1
                        self.counted_ids.add(track_id)
                        # Visual Feedback
                        cv2.circle(annotated_frame, (int(cx), int(cy)), 10, (0, 255, 0), -1)
                    else:
                        # Already Counted
                        cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                else:
                    # Outside Zone
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # Draw Total Count
        cv2.putText(annotated_frame, f"Live Count: {self.total_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

    def process_video(self, video_path, output_path, mode="static", on_update=None):
        """
        Processes a video file to count jute bags.
        mode: "static" (whole frame) or "scanning" (center zone)
        """
        import numpy as np # Ensure numpy is available
        print(f"Starting video processing: {video_path} in mode: {mode}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {"count": 0, "status": "failed"}

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video Info: {width}x{height} @ {fps}fps")

        # Output saver
        # Browser-compatible codec (H.264 / avc1)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1 failed")
        except Exception:
            print("Warning: H.264 (avc1) codec failed. Falling back to mp4v.")
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
            
            annotated_frame = frame.copy()

            if mode == "static":
                # --- STATIC MODE: Tiled Detection (SAHI-lite) ---
                # 1. Detect using tiles
                final_boxes, _ = self.detect_with_tiling(frame)
                
                # 2. Update Count (Use High-Water Mark approach for piles)
                # We assume the user is showing the *same* pile, so the best frame is the one with MOST bags.
                snapshot_count = len(final_boxes)
                if snapshot_count > current_count:
                    current_count = snapshot_count
                    if on_update:
                         try: on_update({"count": self.total_count + current_count, "frame_idx": frame_idx}) 
                         except: pass

                # 3. Optimize Visualization (Static)
                cv2.putText(annotated_frame, "STATIC MODE - TILED SCAN", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for box in final_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Draw Box & Dot
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)

            else:
                # --- SCANNING MODE: Center Zone Tracking ---
                # Run YOLOv8 tracking with OPTIMIZED parameters
                results = self.model.track(frame, persist=True, conf=0.6, iou=0.5, 
                                        tracker="bytetrack.yaml", 
                                        agnostic_nms=True,
                                        classes=[0],
                                        verbose=False)
                
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    detection_count += 1
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                    
                    annotated_frame = results[0].plot() # Use default plot for tracking debug

                    # --- SCANNING MODE (Center Zone) ---
                    # Box in the middle 60% of the screen
                    zone_x1 = int(width * 0.2)
                    zone_x2 = int(width * 0.8)
                    zone_y1 = int(height * 0.1)
                    zone_y2 = int(height * 0.9)
                    
                    # Draw Zone (Blue Box)
                    cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 255, 0), 2)
                    cv2.putText(annotated_frame, "SCANNING ZONE", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        cx, cy = float(x), float(y)
                        
                        # Check if center of bag is inside the zone
                        is_in_zone = (zone_x1 < cx < zone_x2) and (zone_y1 < cy < zone_y2)
                        
                        if is_in_zone:
                            if track_id not in counted_ids:
                                # NEW VALID BAG
                                current_count += 1
                                counted_ids.add(track_id)
                                cv2.circle(annotated_frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                                cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                                if on_update:
                                    try:
                                        on_update({"count": self.total_count + current_count, "frame_idx": frame_idx})
                                    except:
                                        pass
                            else:
                                # ALREADY COUNTED
                                cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                        else:
                            # OUTSIDE ZONE
                            cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            # Draw counting info
            mode_label = "Scanner" if mode == "scanning" else "Static (Max)"
            cv2.putText(annotated_frame, f"Total Bags ({mode_label}): {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write merged frame
            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()
        
        # Update global total
        self.total_count += current_count 
        
        print(f"Processed video saved to {output_path} | Final Count: {current_count}")
        return {"count": current_count, "status": "completed"}

    # Generator for future streaming support
    # def process_video_generator(self, video_path, line_y=500):
    #     ... implementation deferred ...
