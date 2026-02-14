import cv2
import numpy as np
import torch

def get_centroid(mask_logits):
    """
    Calculate centroid (cx, cy) from mask logits.
    """
    # Convert logits to binary mask (threshold > 0.0)
    mask = (mask_logits > 0.0).float()
    
    if mask.sum() == 0:
        return None

    # Calculate centroid using moments or simple mean of coordinates
    # shape is (1, H, W) or (H, W)
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
        
    h, w = mask.shape
    y_indices, x_indices = torch.where(mask > 0)
    
    if len(y_indices) == 0:
        return None
        
    cy = y_indices.float().mean().item()
    cx = x_indices.float().mean().item()
    
    return (cx, cy)

def annotate_frame(frame, detections, total_count, line_y=500):
    """
    Draw masks, boxes, and IDs on the frame using OpenCV.
    Expects detections to be a dict or similar structure.
    """
    if not isinstance(detections, dict):
        return frame

    obj_ids = detections.get("obj_ids", [])
    mask_logits = detections.get("mask_logits", [])

    for i, obj_id in enumerate(obj_ids):
        # Draw Mask
        if i < len(mask_logits):
            mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
            if mask.ndim == 2:
                # Create colored mask overlay
                color = ((obj_id * 50) % 255, (obj_id * 100) % 255, (obj_id * 150) % 255)
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0] = color
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
        
        # Draw ID (at centroid)
        centroid = get_centroid(mask_logits[i])
        if centroid:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.putText(frame, f"ID: {obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw Global Count
    cv2.putText(frame, f"Total Count: {total_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw Line
    h, w, _ = frame.shape
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 2)
    
    return frame
