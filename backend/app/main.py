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
from .zone_tracker import ModularZoneTracker

# Global tracker placeholders
tracker = None
zone_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    global tracker, zone_tracker
    
    use_mock = os.getenv("USE_MOCK_TRACKER", "false").lower() == "true"
    
    if use_mock:
        print("Starting in MOCK / SIMULATION MODE...")
        from .mock_tracker import MockJuteBagTracker
        tracker = MockJuteBagTracker()
    else:
        print("Initializing JuteBagTracker...")
        try:
            tracker = JuteBagTracker()
            zone_tracker = ModularZoneTracker()
        except Exception as e:
            print(f"Failed to initialize Real Tracker: {e}")
            print("Falling back to MOCK MODE due to initialization failure.")
            from .mock_tracker import MockJuteBagTracker
            tracker = MockJuteBagTracker()
            zone_tracker = MockJuteBagTracker() # Reuse for simplicity
            
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

app = FastAPI(lifespan=lifespan, title="CCTV VisionCount AI")

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

# --- GLOBAL STATE ---
tasks = {}
# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION_DIR = os.path.join(BASE_DIR, "detections")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # New upload directory
DATA_DIR = os.path.join(BASE_DIR, "data") # Directory for persistent data
TASK_FILE = os.path.join(DATA_DIR, "tasks.json")

def load_tasks():
    global tasks
    if os.path.exists(TASK_FILE):
        try:
            with open(TASK_FILE, "r") as f:
                tasks = json.load(f)
        except Exception as e:
            print(f"Error loading tasks: {e}")
            tasks = {} # Reset tasks if loading fails

def save_tasks():
    try:
        with open(TASK_FILE, "w") as f:
            json.dump(tasks, f, indent=4)
    except Exception as e:
        print(f"Error saving tasks: {e}")

# Ensure directories exist
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists

load_tasks() # Initialize on startup
TEMP_DIR = "backend/temp_uploads" # Use the correct path relative to root if running from root


# Mount static files for video download (Now points to detections folder)
app.mount("/download", StaticFiles(directory=DETECTION_DIR), name="download")

def process_video_task(task_id: str, video_path: str, mode: str = "static"):
    """
    Background task to process video and update status.
    """
    global tracker, zone_tracker
    if not tracker or (mode == "zone" and not zone_tracker):
        print("Tracker(s) not initialized!")
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized"}
        save_tasks()
        return

    print(f"Starting task {task_id} for {video_path} in mode {mode}")
    
    # Callback for real-time updates with persistence
    def safe_broadcast(data: dict):
        # Update persistent task store if progress/count is available
        if task_id in tasks:
            if "progress" in data:
                tasks[task_id]["progress"] = data["progress"]
            if "count" in data:
                tasks[task_id]["results_count"] = data["count"]
            save_tasks()
        
        try:
            asyncio.run(manager.broadcast(data))
        except:
            pass
        
    try:
        # Save output to detections folder with a clean name
        output_filename = f"detected_{task_id}.mp4"
        output_video_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Run tracking and save video with callback
        # v5: Modular Choice between Tracking types
        if mode == "zone":
            zone_tracker.reset_state() # v10.6 Fix: Prevent count leakage across videos
            results = zone_tracker.process_video(video_path, output_video_path, on_update=safe_broadcast)
        else:
            tracker.reset_state() # v10.6 Fix: Standardize reset for all modes
            results = tracker.process_video(video_path, output_video_path, mode=mode, on_update=safe_broadcast)
        
        # Results now contains the count directly from the tracker
        final_count = results.get("count", 0)
        cumulative_total = results.get("total_count", 0) if mode == "zone" else 0
        
        # v8.6 reporting: Use cumulative total for upload status list
        reported_count = cumulative_total if mode == "zone" else final_count
        
        # Force a final broadcast of the global total to ensure UI is in sync
        # v13.0 Precision Fix: Broadcast ONLY the current task's count.
        # This prevents the Summation Bug (6 bag bug)
        safe_broadcast({"count": reported_count})
        
        tasks[task_id] = {
            "status": "completed",
            "count": reported_count,
            "results_count": reported_count,
            "video_url": f"/download/{output_filename}"
        }
        save_tasks()
        
        # Optional: Clean up input file after processing
        # if os.path.exists(video_path):
        #     os.remove(video_path)
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

def process_image_task(task_id: str, image_path: str):
    """
    Background task to process an image.
    """
    global tracker
    if not tracker:
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized"}
        save_tasks()
        return

    print(f"Starting image task {task_id} for {image_path}")
    
    # Callback for real-time updates
    def safe_broadcast(data: dict):
        asyncio.run(manager.broadcast(data))
    
    try:
        output_filename = f"detected_{task_id}.jpg"
        output_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Run processing with callback
        results = tracker.process_image(image_path, output_path, on_update=safe_broadcast)
        
        # Add to task results
        results["video_url"] = f"/download/{output_filename}" # Frontend expects video_url for display
        results["is_image"] = True # Flag for frontend
        tasks[task_id] = results
        save_tasks()
        asyncio.run(manager.broadcast({"count": tracker.total_count}))
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: str = Form("static")):
    """
    Uploads a file (Video or Image) and starts processing.
    """
    # Generate unique ID
    task_id = str(uuid.uuid4())
    filename = file.filename.lower()
    
    # Save file
    file_location = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    
    # Determine type
    is_image = filename.endswith(('.jpg', '.jpeg', '.png', '.webp'))
    
    # Validation based on Mode
    if mode == "static" and not is_image:
        return JSONResponse(status_code=400, content={"message": "Static Mode strictly supports IMAGES only (JPG, PNG). Please upload an image."})
    
    if mode == "scanning" and is_image:
        return JSONResponse(status_code=400, content={"message": "Scanning Mode supports VIDEOS only. Please upload a video."})

    if mode == "zone" and is_image:
        return JSONResponse(status_code=400, content={"message": "Zone Mode supports VIDEOS only. Please upload a video."})

    # v13.0 Critical Reset: Standardize clean slate for ALL trackers
    if tracker: tracker.reset_state()
    if zone_tracker: zone_tracker.reset_state()
    
    # Initial task status
    tasks[task_id] = {"status": "processing", "progress": 0, "file": file.filename, "mode": mode}
    save_tasks()
    
    # Start background processing
    if is_image:
        background_tasks.add_task(process_image_task, task_id, file_location)
    else:
        background_tasks.add_task(process_video_task, task_id, file_location, mode)
    
    return {"task_id": task_id, "message": "Upload accepted and processing started."}

@app.post("/reset")
async def reset_session():
    """Resets the session count and history."""
    global tracker, zone_tracker
    if tracker:
        tracker.reset_state()
    if zone_tracker:
        try:
            zone_tracker.reset_state()
        except Exception as e:
            print(f"Zone reset failed: {e}")
            
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
