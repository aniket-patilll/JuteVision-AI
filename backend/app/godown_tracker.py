"""
Godown Counting/Decounting Tracker
===================================
Direction-aware sack counting for warehouse (godown) entry/exit monitoring.
Uses a configurable horizontal counting line — sacks crossing downward = IN (+1),
sacks crossing upward = OUT (-1). Persistent inventory stored in JSON.
"""

import cv2
import torch
import numpy as np
import os
import json
import time
from ultralytics import YOLO


class GodownTracker:
    def __init__(self, model_name="sacks_custom.pt"):
        print("Initializing GodownTracker...")

        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Load YOLO sacks model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), "models")
        model_path = os.path.join(models_dir, model_name)

        try:
            self.model = YOLO(model_path)
            print(f"GodownTracker: Sacks model loaded from {model_path}")
        except Exception as e:
            print(f"GodownTracker: Error loading sacks model: {e}")
            self.model = None

        # Load standard YOLO model for PERSON detection
        person_model_path = os.path.join(models_dir, "yolov8n.pt")
        try:
            self.person_model = YOLO(person_model_path)
            print(f"GodownTracker: Person model loaded from {person_model_path}")
        except Exception as e:
            print(f"GodownTracker: Person model not available: {e}")
            self.person_model = None

        # Persistent inventory file
        self.data_dir = os.path.join(os.path.dirname(current_dir), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.inventory_file = os.path.join(self.data_dir, "godown_inventory.json")

        # State
        self.reset_state()

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def reset_state(self):
        """Resets the tracker counting state (not inventory)."""
        print("Resetting GodownTracker state...")
        self.track_positions = {}  # tid -> list of (cx, cy, frame_idx)
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.today_in = 0
        self.today_out = 0
        self.events = []  # Rolling event log
        self.display_id_counter = 0
        self.tid_to_display_id = {}
        self.person_carry_history = {} # Tracks if person has been carrying recently
        self.max_initial_sacks = 0 # Tracks initial godown inventory max count during scan
        self.initial_right_tids = {} # Tracks how many frames a tid has spent on the right side initially
        self.person_tainted_ids = set()  # Sack track IDs that overlapped with a person (permanently banned)
        return {"status": "reset"}

    # --- Persistent Inventory ---

    def load_inventory(self):
        """Load inventory count from persistent JSON file."""
        try:
            if os.path.exists(self.inventory_file):
                with open(self.inventory_file, "r") as f:
                    data = json.load(f)
                    return data.get("inventory", 0)
        except Exception as e:
            print(f"GodownTracker: Error loading inventory: {e}")
        return 0

    def save_inventory(self, inventory):
        """Save inventory count to persistent JSON file."""
        try:
            data = {
                "inventory": inventory,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.inventory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"GodownTracker: Error saving inventory: {e}")

    def set_baseline(self, count):
        """Manually set the inventory baseline (e.g., 500 sacks already in godown)."""
        self.save_inventory(count)
        return {"status": "baseline_set", "inventory": count}

    def get_status(self):
        """Returns current godown status."""
        inventory = self.load_inventory()
        return {
            "inventory": inventory,
            "today_in": self.today_in,
            "today_out": self.today_out,
            "net_change": self.today_in - self.today_out,
        }

    def reset_daily(self):
        """Reset daily in/out counters (called at start of day or manually)."""
        self.today_in = 0
        self.today_out = 0
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.events = []
        return {"status": "daily_reset", "inventory": self.load_inventory()}

    # --- Direction Detection ---

    def _get_display_id(self, tid):
        """Assigns a sequential display ID to a tracker ID."""
        if tid not in self.tid_to_display_id:
            self.display_id_counter += 1
            self.tid_to_display_id[tid] = self.display_id_counter
        return self.tid_to_display_id[tid]

    def _detect_crossing(self, tid, line_x, frame_idx):
        """
        Detect if a tracked object crossed the counting line and in which direction.
        Returns: 'in', 'out', or None
        """
        positions = self.track_positions.get(tid, [])
        if len(positions) < 2:
            return None

        # Use a wider span of recent positions for more reliable direction
        prev_positions = positions[-10:]  # Last 10 positions
        
        # Check intermediate positions for line crossing
        crossed_left_to_right = False
        crossed_right_to_left = False
        for i in range(1, len(prev_positions)):
            px, py, pf = prev_positions[i - 1]
            cx, cy, cf = prev_positions[i]
            
            # Prevent fake crossings caused by ID jumping across the screen
            # If the centroid jumps an impossible distance between tracking frames, ignore it
            frames_elapsed = max(1, cf - pf)
            speed_x = abs(cx - px) / frames_elapsed
            if speed_x > 50: # 50 pixels per frame purely on X axis is insanely fast for a carried sack
                continue
                
            # Left to Right = IN
            if px < line_x and cx >= line_x:
                crossed_left_to_right = True
            # Right to Left = OUT
            if px > line_x and cx <= line_x:
                crossed_right_to_left = True

        if crossed_left_to_right and tid not in self.counted_in_ids:
            self.counted_in_ids.add(tid)
            return "in"
        
        # NOTE: User explicitly requested to completely disable 'OUT' tracking in Godown mode
        # so that inventory never decreases and out count remains permanently Zero.
        # elif crossed_right_to_left and tid not in self.counted_out_ids:
        #     self.counted_out_ids.add(tid)
        #     return "out"

        return None

    # --- Video Processing ---

    def process_video(self, video_path, output_path, line_position=0.5, on_update=None):
        """
        Process a video file with direction-based counting.
        line_position: 0.0 to 1.0 (fraction of frame height for counting line)
        """
        self.reset_state()

        if self.model is None:
            return {"count": 0, "status": "model_not_loaded"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"count": 0, "status": "failed_to_open_video"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        # Counting line X position
        line_x = int(width * line_position)

        # Output video writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1 failed")
        except Exception:
            print("GodownTracker: H.264 codec failed, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset inventory for this video session (fresh start)
        self.save_inventory(0)
        inventory = 0
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame = frame.copy()

            # --- PERSON DETECTION FIRST (to filter sack false positives) ---
            person_boxes_frame = []  # Store person bounding boxes (x1,y1,x2,y2)
            person_centroids_frame = []  # Store person centroids (cx,cy,h) for proximity check
            if self.person_model is not None:
                person_results = self.person_model.track(
                    frame,
                    persist=True,
                    conf=0.10,  # Very low to catch all workers reliably
                    iou=0.5,
                    tracker="bytetrack.yaml",
                    classes=[0],  # Person class (COCO)
                    verbose=False,
                )

                if person_results and person_results[0].boxes.id is not None:
                    p_boxes = person_results[0].boxes.xywh.cpu().numpy()
                    p_ids = person_results[0].boxes.id.int().cpu().tolist()

                    for box, tid in zip(p_boxes, p_ids):
                        cx, cy = float(box[0]), float(box[1])
                        w, h = float(box[2]), float(box[3])
                        x1 = int(cx - w / 2)
                        y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)
                        person_boxes_frame.append((x1, y1, x2, y2))
                        person_centroids_frame.append((cx, cy, h))  # Store centroid + height

                        # Draw person box (gray, non-counting)
                        person_tid = tid + 100000
                        display_id = self._get_display_id(person_tid)
                        color = (150, 150, 150)
                        label = f"W:{display_id}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(annotated_frame, (int(cx), int(cy)), 4, color, -1)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            # Run YOLO tracking for SACKS
            results = self.model.track(
                frame,
                persist=True,
                conf=0.10,  # Low enough to detect sacks in all videos
                iou=0.5,
                tracker="bytetrack.yaml",
                classes=[0],  # Sacks only
                verbose=False,
            )

            if frame_idx < 60 and results and results[0].boxes.id is not None:
                boxes_scan = results[0].boxes.xywh.cpu().numpy()
                ids_scan = results[0].boxes.id.int().cpu().tolist()
                current_valid_boxes = 0
                valid_tids_this_frame = []
                for box_s, tid_s in zip(boxes_scan, ids_scan):
                    scx, scy = float(box_s[0]), float(box_s[1])
                    sw, sh = float(box_s[2]), float(box_s[3])
                    
                    # We want to count ALL visible static valid sacks in the godown 
                    # for the initial baseline inventory, regardless of which side of the line they are on.
                    # Relaxed filters for godown
                    sar = sw / sh if sh > 0 else 1
                    if sar < 0.2 or sar > 6.0:
                        continue
                    if sw < width * 0.005 or sh < height * 0.005:
                        continue
                    if sw > width * 0.85 or sh > height * 0.85:
                        continue
                        
                    # Skip human-shaped detections
                    if sar < 0.65 and sh > height * 0.20:
                        self.person_tainted_ids.add(tid_s)
                        continue
                        
                    # Large area filter (A single sack is never more than ~12% of the frame)
                    frame_area = max(1, width * height)
                    if (sw * sh) / frame_area > 0.12:
                        self.person_tainted_ids.add(tid_s)
                        continue
                        
                    if tid_s in self.person_tainted_ids:
                        continue
                    
                    # Sacks are carried on people's backs, so overlap is guaranteed.
                    # We rely purely on the aspect ratio shape filter (sar < 0.65) above.
                    
                    current_valid_boxes += 1
                    valid_tids_this_frame.append(tid_s)
                
                # If we see more valid sacks in this frame than before across the WHOLE screen, use it as baseline
                if current_valid_boxes > self.max_initial_sacks:
                    diff = current_valid_boxes - self.max_initial_sacks
                    self.max_initial_sacks = current_valid_boxes
                    
                    # DO NOT increment self.today_in. Initial baseline is just existing inventory.
                    inventory += diff
                    self.save_inventory(inventory)
                    
                    for t in valid_tids_this_frame:
                        self.counted_in_ids.add(t)
                        
                    print(f"GodownTracker: Initial scan max baseline updated to {self.max_initial_sacks}. Diff: +{diff}")
                    
                    if on_update:
                        try:
                            # We send an event without triggering godown_in so today_in isn't falsely inflated
                            on_update({
                                "event": "baseline_update",
                                "inventory": inventory,
                                "today_in": self.today_in,
                                "today_out": self.today_out,
                            })
                        except:
                            pass

            # --- Draw Counting Line ---
            cv2.line(annotated_frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
            # Text drawn sideways along the line
            cv2.putText(
                annotated_frame,
                "COUNTING LINE",
                (line_x - 15, height // 2 - 80), # rotated using standard putText rotated requires getRotationMatrix, so we'll just write it normally
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Direction labels
            cv2.putText(
                annotated_frame,
                "IN (Right) ->",
                (line_x + 10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                "<- OUT (Left)",
                (line_x - 130, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()

                for box, tid in zip(boxes, ids):
                    cx, cy = float(box[0]), float(box[1])
                    w, h = float(box[2]), float(box[3])

                    # Relaxed filters for godown (allow small far sacks & person+sack combos)
                    aspect_ratio = w / h if h > 0 else 1
                    if aspect_ratio < 0.2 or aspect_ratio > 6.0:
                        continue
                    if w < width * 0.005 or h < height * 0.005:
                        continue
                    if w > width * 0.85 or h > height * 0.85:
                        continue

                    # --- HUMAN-SHAPE FILTER ---
                    # Workers are typically tall and narrow (aspect_ratio < 0.65)
                    # Sacks are wider/squarish. If detection is very tall & narrow, skip it.
                    if aspect_ratio < 0.65 and h > height * 0.20:
                        self.person_tainted_ids.add(tid)
                        continue

                    # --- LARGE AREA FILTER ---
                    # A single sack is never more than ~12% of the frame
                    frame_area = max(1, width * height)
                    if (w * h) / frame_area > 0.12:
                        self.person_tainted_ids.add(tid)
                        continue

                    # --- PERSON OVERLAP FILTER (PERSISTENT BLOCKLIST) ---
                    # Only applies to IDs banned by shape/size filters above. 
                    # Aggressive overlap checks removed because sacks are carried on backs.
                    if tid in self.person_tainted_ids:
                        continue

                    sack_x1, sack_y1 = int(cx - w / 2), int(cy - h / 2)
                    sack_x2, sack_y2 = int(cx + w / 2), int(cy + h / 2)
                    sack_area = max(1, w * h)

                    # Update trajectory
                    if tid not in self.track_positions:
                        self.track_positions[tid] = []
                    self.track_positions[tid].append((cx, cy, frame_idx))
                    if len(self.track_positions[tid]) > 30:
                        self.track_positions[tid].pop(0)

                    # Check for line crossing
                    crossing = self._detect_crossing(tid, line_x, frame_idx)
                    display_id = self._get_display_id(tid)

                    if crossing == "in":
                        self.today_in += 1
                        inventory += 1
                        self.save_inventory(inventory)
                        self.events.append(
                            {
                                "msg": f"Sack {display_id} Entered (+1)",
                                "color": (0, 255, 0),
                                "frame": frame_idx,
                            }
                        )

                        # Flash effect
                        cv2.circle(
                            annotated_frame, (int(cx), int(cy)), 15, (0, 255, 0), -1
                        )

                        if on_update:
                            try:
                                on_update(
                                    {
                                        "event": "godown_in",
                                        "display_id": display_id,
                                        "inventory": inventory,
                                        "today_in": self.today_in,
                                        "today_out": self.today_out,
                                    }
                                )
                            except:
                                pass

                    elif crossing == "out":
                        self.today_out += 1
                        inventory = max(0, inventory - 1)
                        self.save_inventory(inventory)
                        self.events.append(
                            {
                                "msg": f"Sack {display_id} Left (-1)",
                                "color": (0, 0, 255),
                                "frame": frame_idx,
                            }
                        )

                        # Flash effect
                        cv2.circle(
                            annotated_frame, (int(cx), int(cy)), 15, (0, 0, 255), -1
                        )

                        if on_update:
                            try:
                                on_update(
                                    {
                                        "event": "godown_out",
                                        "display_id": display_id,
                                        "inventory": inventory,
                                        "today_in": self.today_in,
                                        "today_out": self.today_out,
                                    }
                                )
                            except:
                                pass

                    # Draw bounding box and ID
                    is_right = cx > line_x
                    color = (0, 255, 0) if is_right else (0, 150, 255)

                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 4, color, -1)
                    cv2.putText(
                        annotated_frame,
                        f"ID:{display_id}",
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            # --- PERSON DETECTION ---
            # Person detection is now handled BEFORE sack tracking (above)
            # Workers are drawn as gray boxes but NEVER counted as sacks
            # regardless of whether they appear to be carrying

            # --- HUD: Inventory Stats ---
            # Background panel
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 145), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            cv2.putText(
                annotated_frame,
                f"GODOWN INVENTORY: {inventory}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"IN Today: +{self.today_in}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"OUT Today: -{self.today_out}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 100, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Net: {'+' if (self.today_in - self.today_out) >= 0 else ''}{self.today_in - self.today_out}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # --- Event Feed (Top-Right) ---
            recent_events = self.events[-5:]
            for i, event in enumerate(reversed(recent_events)):
                age = frame_idx - event["frame"]
                if age > 100:
                    continue
                y_pos = 50 + (i * 30)
                cv2.putText(
                    annotated_frame,
                    event["msg"],
                    (width - 300, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    event["color"],
                    2,
                )

            # Mode label
            cv2.putText(
                annotated_frame,
                "GODOWN MODE",
                (width - 200, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            out.write(annotated_frame)

            # Broadcast frame
            if on_update and frame_idx % 2 == 0:
                try:
                    import base64

                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
                    on_update(
                        {
                            "type": "frame",
                            "data": jpg_as_text,
                            "count": inventory,
                            "today_in": self.today_in,
                            "today_out": self.today_out,
                        }
                    )
                except Exception as e:
                    print(f"GodownTracker: Frame broadcast failed: {e}")

            frame_idx += 1

        cap.release()
        out.release()

        # Save final inventory
        self.save_inventory(inventory)

        print(
            f"GodownTracker: Completed. Inventory={inventory}, In={self.today_in}, Out={self.today_out}"
        )

        return {
            "count": inventory,
            "today_in": self.today_in,
            "today_out": self.today_out,
            "net_change": self.today_in - self.today_out,
            "status": "completed",
        }

    # --- Live Frame Processing ---

    def process_live_frame(self, frame, line_position=0.5, on_update=None):
        """
        Process a single frame for live godown CCTV monitoring.
        Uses a vertical counting line (left=OUT, right=IN) consistent with video mode.
        Returns annotated frame.
        """
        if self.model is None:
            return frame

        height, width = frame.shape[:2]
        line_x = int(width * line_position)
        annotated_frame = frame.copy()

        # Use incremental frame index for live mode
        if not hasattr(self, '_live_frame_idx'):
            self._live_frame_idx = 0
        self._live_frame_idx += 1
        frame_idx = self._live_frame_idx

        inventory = self.load_inventory()

        # --- PERSON DETECTION FIRST ---
        person_boxes_frame = []
        person_centroids_frame = []
        if self.person_model is not None:
            person_results = self.person_model.track(
                frame, persist=True, conf=0.10, iou=0.5,
                tracker="bytetrack.yaml", classes=[0], verbose=False,
            )
            if person_results and person_results[0].boxes.id is not None:
                p_boxes = person_results[0].boxes.xywh.cpu().numpy()
                p_ids = person_results[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(p_boxes, p_ids):
                    cx, cy = float(box[0]), float(box[1])
                    w, h = float(box[2]), float(box[3])
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    person_boxes_frame.append((x1, y1, x2, y2))
                    person_centroids_frame.append((cx, cy, h))
                    person_tid = tid + 100000
                    display_id = self._get_display_id(person_tid)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
                    cv2.putText(annotated_frame, f"W:{display_id}", (int(cx)+10, int(cy)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # --- SACK DETECTION ---
        results = self.model.track(
            frame, persist=True, conf=0.10, iou=0.5,
            tracker="bytetrack.yaml", classes=[0], verbose=False,
        )

        # Draw counting line
        cv2.line(annotated_frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
        cv2.putText(annotated_frame, "COUNTING LINE", (line_x - 15, height // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "IN (Right) ->", (line_x + 10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        cv2.putText(annotated_frame, "<- OUT (Left)", (line_x - 130, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, ids):
                cx, cy = float(box[0]), float(box[1])
                w, h = float(box[2]), float(box[3])

                aspect_ratio = w / h if h > 0 else 1
                if aspect_ratio < 0.2 or aspect_ratio > 6.0: continue
                if w < width * 0.005 or h < height * 0.005: continue
                if w > width * 0.85 or h > height * 0.85: continue

                if aspect_ratio < 0.65 and h > height * 0.20:
                    self.person_tainted_ids.add(tid)
                    continue

                # Large area filter
                frame_area = max(1, width * height)
                if (w * h) / frame_area > 0.12:
                    self.person_tainted_ids.add(tid)
                    continue

                if tid in self.person_tainted_ids:
                    continue

                sack_x1, sack_y1 = int(cx - w / 2), int(cy - h / 2)
                sack_x2, sack_y2 = int(cx + w / 2), int(cy + h / 2)
                sack_area = max(1, w * h)
                is_person = False
                for (pcx, pcy, ph) in person_centroids_frame:
                    dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    if dist < ph * 0.8:
                        is_person = True
                        break
                if not is_person:
                    for (px1, py1, px2, py2) in person_boxes_frame:
                        ix1, iy1 = max(sack_x1, px1), max(sack_y1, py1)
                        ix2, iy2 = min(sack_x2, px2), min(sack_y2, py2)
                        if ix1 < ix2 and iy1 < iy2:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            if intersection / sack_area > 0.10:
                                is_person = True
                                break
                if is_person:
                    self.person_tainted_ids.add(tid)
                    continue

                if tid not in self.track_positions:
                    self.track_positions[tid] = []
                self.track_positions[tid].append((cx, cy, frame_idx))
                if len(self.track_positions[tid]) > 30:
                    self.track_positions[tid].pop(0)

                crossing = self._detect_crossing(tid, line_x, frame_idx)
                display_id = self._get_display_id(tid)

                if crossing == "in":
                    self.today_in += 1
                    inventory += 1
                    self.save_inventory(inventory)
                    self.events.append({"msg": f"Sack {display_id} Entered (+1)", "color": (0, 255, 0), "frame": frame_idx})
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 15, (0, 255, 0), -1)
                    if on_update:
                        try:
                            on_update({"event": "godown_in", "display_id": display_id,
                                       "inventory": inventory, "today_in": self.today_in, "today_out": self.today_out})
                        except: pass

                elif crossing == "out":
                    self.today_out += 1
                    inventory = max(0, inventory - 1)
                    self.save_inventory(inventory)
                    self.events.append({"msg": f"Sack {display_id} Left (-1)", "color": (0, 0, 255), "frame": frame_idx})
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 15, (0, 0, 255), -1)
                    if on_update:
                        try:
                            on_update({"event": "godown_out", "display_id": display_id,
                                       "inventory": inventory, "today_in": self.today_in, "today_out": self.today_out})
                        except: pass

                is_right = cx > line_x
                color = (0, 255, 0) if is_right else (0, 150, 255)
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, (int(cx), int(cy)), 4, color, -1)
                cv2.putText(annotated_frame, f"ID:{display_id}", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- HUD ---
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 145), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        cv2.putText(annotated_frame, f"GODOWN INVENTORY: {inventory}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"IN Today: +{self.today_in}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"OUT Today: -{self.today_out}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        net = self.today_in - self.today_out
        cv2.putText(annotated_frame, f"Net: {'+' if net >= 0 else ''}{net}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for i, event in enumerate(reversed(self.events[-5:])):
            age = frame_idx - event["frame"]
            if age > 300: continue
            cv2.putText(annotated_frame, event["msg"], (width - 300, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, event["color"], 2)

        cv2.putText(annotated_frame, "GODOWN LIVE", (width - 200, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated_frame

