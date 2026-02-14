import cv2
import time
import random

class MockJuteBagTracker:
    def __init__(self):
        print("Initializing MockJuteBagTracker... (Simulation Mode)")
        self.total_count = 0
        self.counted_ids = set()

    def process_video(self, video_path, output_path, line_y=500, on_update=None):
        """
        Simulates processing a video, detecting bags, and updating count.
        """
        print(f"Mock processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        simulated_bags = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simulate processing time
            # time.sleep(0.01) 
            
            # Simulate finding a bag every 30 frames
            if frame_idx % 30 == 0 and frame_idx > 0:
                simulated_bags += 1
                self.total_count += 1
                
                # Draw a fake bounding box
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                cv2.putText(frame, f"Bag {simulated_bags}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Trigger callback
                if on_update:
                    on_update({"count": self.total_count, "frame_idx": frame_idx})
            
            # Draw line
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {self.total_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"Mock processed video saved to {output_path}")
        return {} # Return empty dict as results are not used directly in this mock flow
