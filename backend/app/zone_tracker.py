import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO


class ModularZoneTracker:
    def __init__(self, model_name="sacks_custom.pt", target_class_id=None):
        print("Initializing ModularZoneTracker...")

        self.device = self._get_device()
        self.target_class_id = target_class_id

        # Stability tuning (v11.8 Industrial Standard)
        self.entry_threshold = 3 # v11.8 Ultra-sensitive
        self.exit_threshold = 50 # v10.3 Increased to 2.0s to prevent flickering

        self.object_states = {}
        self.roi_points = None
        self.confirmed_centroids = [] # v12.0 Global Spatio-Temporal Deduplication

        # Cumulative Counting (v8.3)
        self.ids_total_confirmed = set() 
        self.total_count = 0
        
        # v8.9 Event Log (Rolling buffer for +1/-1 alerts)
        self.events = []
        
        # v9.0 ID Mapping (Sack 42 -> Sack 1)
        self.tid_to_display_id = {}
        
        # v9.3 Spatial Persistence (Prevent 2 bags -> 3 counts)
        self.recent_confirmations = [] # List of [cx, cy, frame_idx, display_id]
        
        # v9.4 Global State Sync (Sack 1 Entered -> Sack 1 Left)
        self.display_id_states = {} 

        # Load model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), "models")
        model_path = os.path.join(models_dir, model_name)

        try:
            self.model = YOLO(model_path)
            print(f"YOLOv8 loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # 🔥 Bounding Box Overlap Function (Industrial Accurate)
    def bbox_overlap_ratio(self, box, roi):
        x_center, y_center, w, h = box

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        rx1, ry1 = roi[0]
        rx2, ry2 = roi[2]

        inter_x1 = max(x1, rx1)
        inter_y1 = max(y1, ry1)
        inter_x2 = min(x2, rx2)
        inter_y2 = min(y2, ry2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = w * h

        if box_area <= 0:
            return 0

        return inter_area / box_area

    def reset_state(self):
        """Resets the tracker state."""
        print("Resetting ModularZoneTracker state...")
        self.object_states = {}
        self.ids_total_confirmed = set()
        self.confirmed_centroids = [] # v12.0 Global Spatio-Temporal Deduplication
        self.tid_to_display_id = {}
        self.total_count = 0
        self.events = []
        self.recent_confirmations = []
        self.display_id_states = {}
        return {"status": "reset", "count": 0}

    def process_video(self, video_path, output_path, on_update=None):
        self.reset_state() # v13.5 Fresh Start Per Video
        if self.model is None:
            return {"count": 0, "status": "model_not_loaded"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"count": 0, "status": "failed_to_open_video"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        # v13.0 Adaptive ROI (Noise Isolation)
        # 12% Margins for industrial balance
        x1, y1 = int(width * 0.12), int(height * 0.12)
        x2, y2 = int(width * 0.88), int(height * 0.88)
        self.roi_points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ids_confirmed_inside = set()
        frame_idx = 0

        while True:
            # v8.8 Zero-Latency Visual Counter (Reset every frame)
            current_occupancy = 0
            
            success, frame = cap.read()
            if not success:
                break

            annotated_frame = frame.copy()

            results = self.model.track(
                frame,
                persist=True,
                conf=0.20, # v10.5 Cross-Compat precision (Relaxed)
                iou=0.45, # v10.3 Overlap Buff
                tracker="bytetrack.yaml",
                classes=[0], # Strictly track Sacks only
                verbose=False
            )

            detected_ids = set()

            # Draw ROI
            cv2.polylines(annotated_frame, [self.roi_points], True, (255, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                "COUNTING ZONE (ROI)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            if results and results[0].boxes.id is not None:

                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()

                for box, tid, cls in zip(boxes, ids, classes):

                    # Optional class filtering
                    if self.target_class_id is not None:
                        if cls != self.target_class_id:
                            continue

                    # 🔥 Ultra-Strict Industrial Filter (v8.3)
                    w, h = box[2], box[3]
                    aspect_ratio = w / h
                    cx, cy = float(box[0]), float(box[1])
                    
                    # 1. Shape/AR (0.2 to 5.0) - v9.7 Perspective Buff (Relaxed)
                    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                        continue
                    
                    # 2. Hard Screen Margins (2%) - v8.5 Responsive
                    margin_x = width * 0.02
                    margin_y = height * 0.02
                    if (cx < margin_x) or (cx > width - margin_x) or (cy < margin_y):
                        continue
                        
                    # 3. Hard Ground Cut (98%) - v8.5 Conveyor Base (Relaxed)
                    if (cy + h/2 > height * 0.98):
                        continue
                        
                    # 4. Macro-Noise (85% size limit) - v10.3 Industrial Expansion (Relaxed)
                    if (w > width * 0.85) or (h > height * 0.85):
                        continue
                        
                    # 5. Generic Height limit (70%) - v10.3 Perspective Buff
                    if h > (height * 0.70):
                        continue

                    # 6. Minimum Noise Filter (2% Min-Scale) - v13.0 Industrial Standards (Relaxed)
                    if (w < width * 0.02) or (h < height * 0.02):
                        continue

                    # --- PROXIMITY CENTROID DEDUP (v8.3) ---
                    is_duplicate = False
                    for existing_tid, state in self.object_states.items():
                        if existing_tid == tid: continue
                        dist = np.sqrt((cx - state.get("last_cx", 0))**2 + (cy - state.get("last_cy", 0))**2)
                        if dist < 60: # v11.8 60px Per-Frame Dedup (Balanced)
                            is_duplicate = True
                            break
                    
                    if is_duplicate: continue

                    detected_ids.add(tid)
                    overlap = self.bbox_overlap_ratio(box, self.roi_points)
                    # v10.4 Stabilized ROI: 15% for both Entry/Exit triggers
                    inside = overlap > 0.15
                    
                    if inside:
                        current_occupancy += 1

                    # v11.6 Spatial Inheritance for Unconfirmed IDs (Prevent ID Flip resets)
                    if tid not in self.object_states:
                        best_match_tid = None
                        min_dist = 60 # 60px search radius for inheritance
                        for old_tid, old_state in self.object_states.items():
                            if not old_state.get("confirmed", False):
                                d = np.sqrt((cx - old_state.get("last_cx", 0))**2 + (cy - old_state.get("last_cy", 0))**2)
                                if d < min_dist:
                                    min_dist = d
                                    best_match_tid = old_tid
                        
                        if best_match_tid:
                            # Inherit progress
                            old_data = self.object_states[best_match_tid]
                            self.object_states[tid] = {
                                "inside_frames": old_data["inside_frames"],
                                "outside_frames": 0,
                                "confirmed": False,
                                "last_cx": cx,
                                "last_cy": cy,
                                "start_cx": old_data["start_cx"],
                                "start_cy": old_data["start_cy"],
                                "last_event_frame": old_data["last_event_frame"],
                                "alert_state": old_data["alert_state"]
                            }
                        else:
                            self.object_states[tid] = {
                                "inside_frames": 0,
                                "outside_frames": 0,
                                "confirmed": False,
                                "last_cx": cx,
                                "last_cy": cy,
                                "start_cx": cx, # v10.7 Movement Guard Path Tracking
                                "start_cy": cy, # v10.7 Movement Guard Path Tracking
                                "last_event_frame": -100, # v9.1 Temporal Guard
                                "alert_state": None # v9.2 State Machine Lock
                            }

                    state = self.object_states[tid]
                    state["last_cx"] = cx
                    state["last_cy"] = cy
                    if inside:
                        state["inside_frames"] += 1
                        state["outside_frames"] = 0
                        
                        if state["inside_frames"] >= 2 and not state["confirmed"]: # v14.3 High-Recall Balance
                            # v12.1 Global Spatio-Temporal Precision Logic
                            total_travel = np.sqrt((cx - state["start_cx"])**2 + (cy - state["start_cy"])**2)
                            
                            # v12.11 sensitivity: 5px (Recall motion buffer)
                            if total_travel > 5:
                                # 1. Global Centroid Rejection (v13.0 Precision Absolute)
                                is_reconfirmed = False
                                
                                # Dynamic Radius (High-Recall Balance for 7-sack density):
                                # Large (Conveyor): width * 0.15 (Stable)
                                # Small (Workers): width * 0.06 (Approx 36px)
                                radius = width * 0.15 if (w > width * 0.20) else width * 0.06
                                
                                for old_cx, old_cy, old_frame in self.confirmed_centroids:
                                    dist_to_confirmed = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                                    if dist_to_confirmed < radius and (frame_idx - old_frame) < 100:
                                        is_reconfirmed = True
                                        break
                                
                                if not is_reconfirmed:
                                    # 2. Local ID Jump Protection
                                    matched_display_id = None
                                    # Dynamic Radius (Matching reconfirmation):
                                    jump_radius = width * 0.15 if (w > width * 0.20) else width * 0.06
                                    self.recent_confirmations = [c for c in self.recent_confirmations if frame_idx - c[2] < 100]
                                    for rc_x, rc_y, rc_f, rc_id in self.recent_confirmations:
                                        if np.sqrt((cx - rc_x)**2 + (cy - rc_y)**2) < jump_radius:
                                            matched_display_id = rc_id
                                            break
                                    
                                    if matched_display_id:
                                        # ID JUMP: Map to existing ID
                                        self.tid_to_display_id[tid] = matched_display_id
                                        self.ids_total_confirmed.add(tid)
                                        global_state = self.display_id_states.get(matched_display_id, {"alert_state": None, "confirmed": False})
                                        state["alert_state"] = global_state["alert_state"]
                                        state["confirmed"] = global_state["confirmed"]
                                    else:
                                        # NEW BAG: Register
                                        self.ids_total_confirmed.add(tid)
                                        self.total_count += 1
                                        display_id = self.total_count
                                        self.tid_to_display_id[tid] = display_id
                                        self.display_id_states[display_id] = {
                                            "alert_state": None, 
                                            "confirmed": False,
                                            "start_cx": cx,
                                            "start_cy": cy
                                        }
                                        self.recent_confirmations.append([cx, cy, frame_idx, display_id])
                                        self.confirmed_centroids.append((cx, cy, frame_idx))
                                    
                                    display_id = self.tid_to_display_id[tid]
                                    global_data = self.display_id_states.get(display_id, {})
                                    if global_data.get("alert_state") != "entered":
                                        state["alert_state"] = "entered"
                                        self.display_id_states[display_id]["alert_state"] = "entered"
                                        self.display_id_states[display_id]["confirmed"] = True
                                        state["last_event_frame"] = frame_idx
                                        self.events.append({
                                            "msg": f"Sack {display_id} Entered (+1)",
                                            "color": (0, 255, 0),
                                            "frame": frame_idx
                                        })

                        color = (0, 255, 0)

                    else:
                        state["outside_frames"] += 1
                        state["inside_frames"] = 0

                        if (
                            state["outside_frames"] >= self.exit_threshold
                            and state["confirmed"]
                        ):
                            display_id = self.tid_to_display_id.get(tid, tid)
                            global_data = self.display_id_states.get(display_id, {})
                            
                            # v9.4 Global Guard: Only trigger "Left" if the Display ID is currently "Entered"
                            if global_data.get("alert_state") == "entered":
                                state["confirmed"] = False
                                state["alert_state"] = "left"
                                self.display_id_states[display_id]["alert_state"] = "left"
                                self.display_id_states[display_id]["confirmed"] = False
                                state["last_event_frame"] = frame_idx
                                
                                # v8.9 Exit Event (-1)
                                self.events.append({
                                    "msg": f"Sack {display_id} Left (-1)",
                                    "color": (0, 0, 255),
                                    "frame": frame_idx
                                })
                            
                            ids_confirmed_inside.discard(tid)

                        color = (0, 0, 255)

                    # Draw bounding box center for visualization
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 5, color, -1)
                    
                    # v9.0 Show Sequential ID if available
                    display_id = self.tid_to_display_id.get(tid, tid)
                    cv2.putText(
                        annotated_frame,
                        f"ID:{display_id}",
                        (int(cx), int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

            # Handle disappeared IDs
            for tid in list(self.object_states.keys()):
                if tid not in detected_ids:
                    self.object_states[tid]["outside_frames"] += 1

                    if self.object_states[tid]["outside_frames"] >= self.exit_threshold:
                        # v9.4 Global Secondary Cleanup
                        display_id = self.tid_to_display_id.get(tid, tid)
                        global_data = self.display_id_states.get(display_id, {})
                        
                        if global_data.get("alert_state") == "entered":
                             self.display_id_states[display_id]["alert_state"] = "left"
                             self.display_id_states[display_id]["confirmed"] = False
                             self.events.append({
                                "msg": f"Sack {display_id} Left (-1)",
                                "color": (0, 0, 255),
                                "frame": frame_idx
                            })
                        ids_confirmed_inside.discard(tid)
                        del self.object_states[tid]

            # v8.8: live_count now uses the immediate visual occupancy for the UI
            live_count = current_occupancy

            cv2.putText(
                annotated_frame,
                f"Sacks in ROI: {live_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # --- v8.9 DRAW EVENT FEED (Top-Right) ---
            recent_events = self.events[-5:] # Show last 5 events
            for i, event in enumerate(reversed(recent_events)):
                # Fade out older events (frame-based)
                age = frame_idx - event["frame"]
                if age > 100: continue # Expire after 100 frames
                
                y_pos = 50 + (i * 30)
                cv2.putText(
                    annotated_frame,
                    event["msg"],
                    (width - 300, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    event["color"],
                    2
                )

            out.write(annotated_frame)

            # v8.7 High-Frequency Broadcast (Every 2 frames)
            if on_update and frame_idx % 2 == 0:
                import base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                on_update({
                    "type": "frame",
                    "data": jpg_as_text,
                    "count": live_count
                })

            frame_idx += 1

        cap.release()
        out.release()

        return {
            "count": live_count, 
            "total_count": self.total_count, 
            "status": "completed"
        }
