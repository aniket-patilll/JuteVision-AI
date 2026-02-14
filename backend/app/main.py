from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
import shutil
import os
import cv2
import uuid
import json
import asyncio
from .tracker import JuteBagTracker

# Global tracker placeholder
tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    global tracker
    
    use_mock = os.getenv("USE_MOCK_TRACKER", "false").lower() == "true"
    
    if use_mock:
        print("Starting in MOCK / SIMULATION MODE...")
        from .mock_tracker import MockJuteBagTracker
        tracker = MockJuteBagTracker()
    else:
        print("Initializing JuteBagTracker...")
        try:
            tracker = JuteBagTracker()
        except Exception as e:
            print(f"Failed to initialize Real Tracker: {e}")
            print("Falling back to MOCK MODE due to initialization failure.")
            from .mock_tracker import MockJuteBagTracker
            tracker = MockJuteBagTracker()
            
    yield
    # Clean up on shutdown if needed
    print("Shutting down JuteBagTracker...")
    tracker = None

from fastapi.staticfiles import StaticFiles

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message: {e}")

manager = ConnectionManager()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        if tracker:
            await websocket.send_json({"count": tracker.total_count})
            
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# In-memory task store
tasks = {}
TEMP_DIR = "backend/temp_uploads" # Use the correct path relative to root if running from root
# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION_DIR = os.path.join(BASE_DIR, "detections")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # New upload directory

# Ensure directories exist
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files for video download (Now points to detections folder)
app.mount("/download", StaticFiles(directory=DETECTION_DIR), name="download")

def process_video_task(task_id: str, video_path: str, mode: str = "static"):
    """
    Background task to process video and update status.
    """
    global tracker
    if not tracker:
        print("Tracker not initialized!")
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized"}
        return

    print(f"Starting task {task_id} for {video_path} in mode {mode}")
    
    # Callback for real-time updates
    def on_update(data: dict):
        import asyncio
        # We need to run the async broadcast from this sync callback
        def safe_broadcast(data: dict):
             asyncio.run(manager.broadcast(data))
        try:
             safe_broadcast(data)
        except Exception as e:
            print(f"WS Broadcast error: {e}")

    # Simplified Async Wrapper for the callback if running in thread
    def safe_broadcast(data: dict):
        asyncio.run(manager.broadcast(data))

    try:
        # Save output to detections folder with a clean name
        output_filename = f"detected_{task_id}.mp4"
        output_video_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Run tracking and save video with callback
        # Default to "static" mode for warehouse videos (counts all unique bags)
        # Use "conveyor" mode for moving belt scenarios
        results = tracker.process_video(video_path, output_video_path, mode=mode, on_update=safe_broadcast)
        
        # Results now contains the count directly from the tracker
        final_count = results.get("count", 0)
        
        # Force a final broadcast of the global total to ensure UI is in sync
        safe_broadcast({"count": tracker.total_count})
        
        tasks[task_id] = {
            "status": "completed",
            "count": final_count,
            "results_count": final_count, # results_count is redundant but kept for compatibility
            "video_url": f"/download/{output_filename}"
        }
        
        # Optional: Clean up input file after processing
        # if os.path.exists(video_path):
        #     os.remove(video_path)
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: str = Form("static")):
    """
    Uploads a video file and starts background processing with the selected mode.
    """
    # Generate unique ID
    task_id = str(uuid.uuid4())
    
    # Save file
    file_location = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read()) # Use await file.read() for async
    
    # Initial task status
    tasks[task_id] = {"status": "processing", "file": file.filename, "mode": mode}
    
    # Start background processing
    background_tasks.add_task(process_video_task, task_id, file_location, mode)
    
    return {"task_id": task_id, "message": "Video uploaded and processing started."}

@app.post("/reset")
async def reset_session():
    """Resets the session count and history."""
    global tracker
    if tracker:
        tracker.reset_state()
        # Broadcast reset to all clients
        await manager.broadcast({"count": 0, "event": "reset"})
    return {"message": "Session reset successfully", "count": 0}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

def generate_frames():
    """
    Generator for camera stream. 
    """
    global tracker
    if not tracker:
        return

    cap = cv2.VideoCapture(0) # Open default camera
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run Live AI Processing
        # This updates the global tracker.total_count
        frame = tracker.process_live_frame(frame)
        
        # Broadcast update to UI every 5 frames to avoid flooding
        # (Optional, but good for keeping the "Total Count" card in sync)
        # if tracker.total_count % 1 == 0: 
        #    ... (requires async magic inside sync generator, skipping for now, stream has visual count)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()

@app.get("/stream")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
