import cv2
import threading

class CameraManager:
    _cap = None
    _lock = threading.Lock()
    
    @classmethod
    def get_cap(cls):
        """Internal method to get the capture object under lock."""
        with cls._lock:
            if cls._cap is None:
                # v13.7 Performance Fix: Optimize for macOS webcam
                cls._cap = cv2.VideoCapture(0)
                if cls._cap is not None and not cls._cap.isOpened():
                    print("CameraManager: ERROR - Failed to open webcam 0")
            return cls._cap

    @classmethod
    def read_frame(cls):
        """Synchronized method to read a frame from the shared camera."""
        with cls._lock:
            if cls._cap is None or not cls._cap.isOpened():
                return False, None
            try:
                success, frame = cls._cap.read()
                return success, frame
            except Exception as e:
                print(f"CameraManager: Read error - {e}")
                return False, None
        
    @classmethod
    def stop(cls):
        """
        Signals to stop reading, but avoids releasing the underlying object 
        immediately to prevent race conditions with active MJPEG threads on Mac.
        """
        with cls._lock:
            # We don't call cls._cap.release() here because it's the primary cause 
            # of segfaults on macOS when other threads are still in a read() call.
            # We just set the state to None to signal it should be treated as stopped.
            cls._cap = None
            print("CameraManager: Session stopping (Camera handle preserved for stability)")
