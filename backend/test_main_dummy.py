import sys
import os
# Add the project root (JuteVision_AI) to sys.path
# dirname is 'backend'. We need '..' to get to project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Mock backend.app.tracker before importing main
mock_tracker_module = MagicMock()
sys.modules["backend.app.tracker"] = mock_tracker_module
mock_tracker_instance = MagicMock()
mock_tracker_instance.total_count = 5
mock_tracker_module.JuteBagTracker.return_value = mock_tracker_instance

# Mock cv2
sys.modules["cv2"] = MagicMock()
# Setup VideoCapture mock
mock_cap = MagicMock()
# Return a few frames then stop
mock_cap.read.side_effect = [(True, b"dummyframe")] * 5 + [(False, None)]
mock_cap.isOpened.return_value = True # ensure loop enters if checked
sys.modules["cv2"].VideoCapture.return_value = mock_cap
# Setup imencode mock
mock_buffer = MagicMock()
mock_buffer.tobytes.return_value = b"encoded_bytes"
sys.modules["cv2"].imencode.return_value = (True, mock_buffer)

# Import app
from backend.app.main import app

client = TestClient(app)

def test_upload_endpoint():
    # Create dummy file
    filename = "test_video.mp4"
    files = {"file": (filename, b"dummy content", "video/mp4")}
    
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["message"] == "Video uploaded and processing started."

def test_stream_endpoint():
    response = client.get("/stream")
    assert response.status_code == 200
    # StreamingResponse is iter, but TestClient handles it
    # We check if headers indicate multipart
    assert "multipart/x-mixed-replace" in response.headers["content-type"]

if __name__ == "__main__":
    test_upload_endpoint()
    test_stream_endpoint()
    print("FastAPI tests passed!")
