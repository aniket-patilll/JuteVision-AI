import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
try:
    from backend.app.utils import get_centroid, annotate_frame
except ImportError:
    from .utils import get_centroid, annotate_frame

class JuteBagTracker:
    def __init__(self, model_name="sacks_custom.pt"):  # Custom sacks model
        print("Initializing JuteBagTracker (Custom Sacks Model)...")
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        # Dynamic path resolution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), "models")
        model_path = os.path.join(models_dir, model_name)
        
        try:
            self.model = YOLO(model_path)
            print(f"YOLOv8 loaded successfully from {model_path}")
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

    def detect_with_tiling(self, frame, strict=False):
        """
        Performs inference using tiling (SAHI-lite) to detect small objects.
        Splits frame into overlapping tiles + full frame, then merges results with NMS.
        """
        if self.model is None:
            return torch.empty((0, 4)), []

        height, width = frame.shape[:2]
        
        # Define Overlapping Tiles (ensure objects on seams are detected)
        # Using a dense 3x3 grid + full frame for maximum coverage
        
        tiles = []
        
        # 1. Full Frame
        tiles.append((0, 0, width, height))
        
        # 2. 2x2 Grid with Overlap
        x_step = int(width * 0.6)
        y_step = int(height * 0.6)
        
        # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        tiles.append((0, 0, x_step, y_step))
        tiles.append((width - x_step, 0, width, y_step))
        tiles.append((0, height - y_step, x_step, height))
        tiles.append((width - x_step, height - y_step, width, height))
        
        # 3. Center Cross (for seams)
        center_w = int(width * 0.6)
        center_h = int(height * 0.6)
        cx_start = int((width - center_w) / 2)
        cy_start = int((height - center_h) / 2)
        tiles.append((cx_start, cy_start, cx_start + center_w, cy_start + center_h))
        
        # 4. Vertical Stripes (Left, Center, Right) for tall piles
        v_w = int(width * 0.4)
        tiles.append((0, 0, v_w, height))
        tiles.append((int(width*0.3), 0, int(width*0.7), height))
        tiles.append((width - v_w, 0, width, height))
        
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
            # v8.1 Balanced Accuracy:
            # - Static Mode (strict=False): High Recall (0.15) for dense piles.
            # - Strict Mode (strict=True): High Precision (0.45) for conveyors/trucks.
            conf_val = 0.45 if strict else 0.15
            results = self.model.predict(tile_img, conf=conf_val, iou=0.60, augment=True, classes=[0], verbose=False)
            
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
        # strict=True: 0.30 (Aggressive anti-ghosting)
        # strict=False: 0.20 (Absolute suppression for dense piles) - v8.3
        nms_thresh = 0.30 if strict else 0.20
        keep_indices = torch.ops.torchvision.nms(all_boxes, all_confs, nms_thresh)
        
        final_boxes = all_boxes[keep_indices]
        final_confs = all_confs[keep_indices]
        
        # --- PROXIMITY-BASED CENTROID DEDUP (v8.3) ---
        # Even with NMS, some boxes vary slightly in coordinates. 
        # We merge boxes whose centers are within 15 pixels.
        deduped_boxes = []
        deduped_confs = []
        
        if len(final_boxes) > 0:
            boxes_np = final_boxes.cpu().numpy()
            confs_np = final_confs.cpu().numpy()
            
            used_mask = np.zeros(len(boxes_np), dtype=bool)
            
            for i in range(len(boxes_np)):
                if used_mask[i]: continue
                
                b1 = boxes_np[i]
                c1_x, c1_y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
                
                # Compare against all subsequent boxes
                for j in range(i + 1, len(boxes_np)):
                    if used_mask[j]: continue
                    
                    b2 = boxes_np[j]
                    c2_x, c2_y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
                    
                    # Euclidean distance between centroids
                    dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                    
                    # 15px Threshold: Absolute zero-double counting guard
                    if dist < 15:
                        used_mask[j] = True # Suppress the lower-confidence duplicate
                
                deduped_boxes.append(torch.tensor(b1))
                deduped_confs.append(torch.tensor(confs_np[i]))
        
        if not deduped_boxes:
             return torch.empty((0, 4)), []
             
        final_boxes = torch.stack(deduped_boxes)
        
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
            # Relaxed for static piles: bags near camera can be large
            is_large = area > frame_area * 0.35 
            is_tiny = area < frame_area * 0.0005
            
            # 2. Shape: Bags are strictly "horizontal/squarish" (0.8 to 2.5). 
            bad_ar = (aspect_ratio < 0.8) or (aspect_ratio > 3.0)
            
            # 3. Position: Top 5% Ceiling rejection
            is_high = cy < (height * 0.05)
            
            # --- v8.1 BALANCED INDUSTRIAL FILTERS ---
            
            # 4. EDGE EXCLUSION MARGINS (v8.4 Balanced)
            # - Strict: 10% (Truck Frame Rejection)
            # - Static: 1% (Allow bags almost to the very edge)
            margin_pct = 0.10 if strict else 0.01
            margin_x = width * margin_pct
            margin_y = height * margin_pct
            is_at_edge = (cx < margin_x) or (cx > width - margin_x) or (cy < margin_y)
            
            # 5. HARD GROUND CUT (v8.1 Balanced)
            # - Strict: 80% (Zero floor noise)
            # - Static: Relaxed, use generic ground filter if at bottom
            is_ground = (y2 > height * 0.80) if strict else (y2 > height * 0.90 and aspect_ratio > 2.5)
            
            # 6. MACRO-NOISE REJECTION (v8.4): No sack is > 45% of screen size in Static
            # Foreground warehouse bags can be massive.
            size_limit = 0.25 if strict else 0.45
            is_too_big = (w > width * size_limit) or (h > height * size_limit)
            
            # 7. TRUCK WALL / PILLAR (Tall & touching side)
            # v8.4: Disable wall filter for static mode to allow edge detections
            touches_side = (x1 < 10) or (x2 > width - 10)
            is_wall = strict and touches_side and (h > height * 0.25)

            if not (is_large or is_tiny or bad_ar or is_high or is_at_edge or is_ground or is_too_big or is_wall):
                valid_boxes.append(box)
                
        if len(valid_boxes) > 0:
            return torch.stack(valid_boxes), list(range(len(valid_boxes)))
        else:
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
        # Relaxed for detection. augment=False for speed in live view.
        results = self.model.track(frame, persist=True, conf=0.3, iou=0.6, 
                                 tracker="bytetrack.yaml", 
                                 agnostic_nms=True,
                                 classes=[0],
                                 augment=False, # Keep false for FPS
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
            return {"count": 0, "status": "failed", "error": f"Could not open video {video_path}"}

        if self.model is None:
             print("Error: Model not loaded")
             return {"count": 0, "status": "failed", "error": "Model not loaded"}

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
                # v8.1: Using balanced (strict=False) for static video piles
                final_boxes, _ = self.detect_with_tiling(frame, strict=False)
                
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
                # augment=True for offline video processing (Robustness)
                results = self.model.track(frame, persist=True, conf=0.15, iou=0.6, 
                                        tracker="bytetrack.yaml", 
                                        agnostic_nms=True,
                                        classes=[0],
                                        augment=True,
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
            
            # Broadcast Frame (Live Feedback)
            if on_update and frame_idx % 2 == 0: # Skip every other frame to save bandwidth if needed
                try:
                    import base64
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    on_update({"type": "frame", "data": jpg_as_text, "count": self.total_count + current_count})
                except Exception as e:
                    print(f"Frame broadcast failed: {e}")

            frame_idx += 1

        cap.release()
        out.release()
        
        # Update global total
        self.total_count += current_count 
        
        print(f"Processed video saved to {output_path} | Final Count: {current_count}")
        return {"count": current_count, "status": "completed"}

    def process_image(self, image_path, output_path, on_update=None):
        """
        Processes a single image file for bag counting.
        """
        import cv2
        import numpy as np

        print(f"Starting image processing: {image_path}")
        
        # Load Image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not open image {image_path}")
            return {"count": 0, "status": "failed", "error": f"Could not open image {image_path}"}
            
        if self.model is None:
             print("Error: Model not loaded")
             return {"count": 0, "status": "failed", "error": "Model not loaded"}
             
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Run Tiled Detection (Best for static piles)
        # v8.1: Using balanced (strict=False) for static image piles
        final_boxes, _ = self.detect_with_tiling(frame, strict=False)
        
        count = len(final_boxes)
        
        # Visualize
        cv2.putText(annotated_frame, f"STATIC IMAGE COUNT: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        for box in final_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw Box & Dot
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)
            
        # Save Output
        cv2.imwrite(output_path, annotated_frame)
        
        # Broadcast Frame (Live Feedback for Image)
        if on_update:
            try:
                import base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                on_update({"type": "frame", "data": jpg_as_text, "count": self.total_count + count})
            except Exception as e:
                print(f"Frame broadcast failed: {e}")

        # Update Global Count
        self.total_count += count
        
        print(f"Processed image saved to {output_path} | Final Count: {count}")
        return {
            "count": count, 
            "status": "completed", 
            "video_url": f"/download/{os.path.basename(output_path)}" # Reuse video_url field for consistency
        }
    # Generator for future streaming support
    # def process_video_generator(self, video_path, line_y=500):
    #     ... implementation deferred ...
