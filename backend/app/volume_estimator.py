import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

class VolumeEstimator:
    def __init__(self, model_name="sacks_custom.pt"):
        print("Initializing VolumeEstimator...")
        self.device = self._get_device()
        
        # Load the YOLO model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), "models")
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            self.model = None
        else:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"VolumeEstimator loaded model on {self.device}")

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _estimate_total_volume(self, boxes, width, height, depth_override=None):
        """
        Geometrically predicts hidden sacks based on visible 2D dimensions.
        """
        if len(boxes) == 0:
            return {"visible_count": 0, "estimated_total": 0, "depth_layers": 0}
            
        visible_count = len(boxes)
        
        # 1. Sack Profiling: Find standard sack dimensions
        widths = [b[2] for b in boxes]
        heights = [b[3] for b in boxes]
        median_w = np.median(widths)
        median_h = np.median(heights)
        
        # 2. Stack Profiling: Find overall bounding box of the whole pile
        min_x = min([b[0] - b[2]/2 for b in boxes])
        max_x = max([b[0] + b[2]/2 for b in boxes])
        min_y = min([b[1] - b[3]/2 for b in boxes])
        max_y = max([b[1] + b[3]/2 for b in boxes])
        
        stack_width = max_x - min_x
        stack_height = max_y - min_y
        
        # Safety checks for tiny/invalid stacks
        if stack_width <= 0 or median_w <= 0:
            return {"visible_count": visible_count, "estimated_total": visible_count, "depth_layers": 1}

        # 3. Depth Estimation Calculation
        depth_override_used = False
        if depth_override is not None and str(depth_override).strip().isdigit():
            estimated_depth_layers = int(depth_override)
            depth_override_used = True
        else:
            # Smart Autonomous Depth Heuristic
            # How many sacks wide and tall is the physical visible stack?
            horizontal_sacks_visible = stack_width / median_w
            vertical_sacks_visible = stack_height / median_h
            
            # Vehicles and industrial stacks (like the truck side-profile)
            # are typically stacked in a pyramid or rectangular prism.
            # If we see a very long horizontal profile (e.g. side of a truck),
            # the depth into the camera is usually narrower (the width of the truck bed ~ 3-4 bags).
            # If we see a very tall profile but narrow width, we are looking at the back of the truck (depth is long).
            
            aspect_ratio = stack_width / max(1, stack_height)
            
            if aspect_ratio > 2.0:
                # Wide shot (Side of the truck/Godown wall). Depth is the Short Edge.
                # Standard truck beds are usually ~8 ft wide, fitting roughly 3 to 4 standard jute bags horizontally.
                estimated_depth_layers = 4 
            elif aspect_ratio < 0.8:
                # Tall/Narrow shot (Back of the truck/Narrow aisle). Depth is the Long Edge.
                # If we are looking at the back, the depth stretches all the way to the front of the truck.
                estimated_depth_layers = max(3, round(vertical_sacks_visible * 1.5))
            else:
                # Square-ish shot (Standard pallet or arbitrary angle).
                # Use a balanced volumetric cube assumption.
                estimated_depth_layers = max(2, round(horizontal_sacks_visible * 0.6))

            # Cap the autonomous logic to prevent astronomical errors on edge cases
            estimated_depth_layers = max(1, min(12, estimated_depth_layers))
        
        # 4. Total Volume Prediction with Packing Efficiency
        packing_efficiency = 0.85 # Accounts for gaps/curved edges
        raw_volume_estimate = visible_count * estimated_depth_layers
        predicted_total = round(raw_volume_estimate * packing_efficiency)
        
        # Sanity Guard: Total cannot be less than what we literally see
        predicted_total = max(visible_count, predicted_total)

        return {
            "visible_count": visible_count,
            "estimated_total": predicted_total,
            "depth_layers": estimated_depth_layers,
            "depth_override_used": depth_override_used
        }

    def process_image(self, image_path, output_path, on_update=None, depth_override=None):
        if self.model is None:
            return {"count": 0, "status": "model_not_loaded"}

        img = cv2.imread(image_path)
        if img is None:
            return {"count": 0, "status": "failed_to_open_image"}

        height, width = img.shape[:2]

        results = self.model(img, conf=0.25)
        
        annotated_img = img.copy()
        boxes_data = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            
            for box in boxes:
                xywh = box.xywh[0].cpu().numpy() # [cx, cy, w, h]
                boxes_data.append(xywh)
                
                # Draw the bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, "Sack", (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Estimate Volume
        estimation = self._estimate_total_volume(boxes_data, width, height)
        
        # Burn info into image
        cv2.putText(annotated_img, f"Visible: {estimation['visible_count']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated_img, f"Est. Depth: {estimation['depth_layers']} Layers", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        cv2.putText(annotated_img, f"Predicted Total: {estimation['estimated_total']}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imwrite(output_path, annotated_img)

        # We return count mapped to 'estimated_total' so it renders elegantly in existing UI logic if needed,
        # but we also return the full payload for the new detailed UI.
        return {
            "status": "completed",
            "count": estimation["estimated_total"], # Main value for backward compatibility list
            "visible_count": estimation["visible_count"],
            "depth_layers": estimation["depth_layers"],
            "estimated_total": estimation["estimated_total"],
            "depth_override_used": estimation["depth_override_used"],
            "estimation_mode": True
        }

    def process_video(self, video_path, output_path, on_update=None, depth_override=None):
        if self.model is None:
            return {"count": 0, "status": "model_not_loaded"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"count": 0, "status": "failed_to_open_video"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        best_estimation = {"visible_count": 0, "estimated_total": 0, "depth_layers": 0, "depth_override_used": False}
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = self.model(frame, conf=0.25, verbose=False)
            annotated_frame = frame.copy()
            boxes_data = []
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                for box in boxes:
                    xywh = box.xywh[0].cpu().numpy()
                    boxes_data.append(xywh)
                    
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Analyze current frame
            current_estimation = self._estimate_total_volume(boxes_data, width, height, depth_override)
            
            # Keep the estimation from the frame that had the MOST visible sacks
            if current_estimation["visible_count"] > best_estimation["visible_count"]:
                best_estimation = current_estimation
                
            # Burn LIVE info into video
            cv2.putText(annotated_frame, f"Visible Sacks Detected: {best_estimation['visible_count']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Current Total Prediction: {best_estimation['estimated_total']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(annotated_frame)
            
            if on_update and frame_idx % 5 == 0:
                import base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                on_update({
                    "type": "frame",
                    "data": jpg_as_text,
                    "count": best_estimation["estimated_total"]
                })
                
            frame_idx += 1

        cap.release()
        out.release()

        return {
            "status": "completed",
            "count": best_estimation["estimated_total"],
            "visible_count": best_estimation["visible_count"],
            "depth_layers": best_estimation["depth_layers"],
            "estimated_total": best_estimation["estimated_total"],
            "depth_override_used": best_estimation.get("depth_override_used", False),
            "estimation_mode": True
        }
