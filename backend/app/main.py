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
from .godown_tracker import GodownTracker
from .volume_estimator import VolumeEstimator
from .multi_camera_tracker import MultiCameraManager
from .video_splitter import split_video

# Global tracker placeholders
tracker = None
zone_tracker = None
godown_tracker = None
volume_estimator = None
multi_cam = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    global tracker, zone_tracker, godown_tracker, volume_estimator, multi_cam
    
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
            godown_tracker = GodownTracker()
            volume_estimator = VolumeEstimator()
            multi_cam = MultiCameraManager()
        except Exception as e:
            print(f"Failed to initialize Real Tracker: {e}")
            print("Falling back to MOCK MODE due to initialization failure.")
            from .mock_tracker import MockJuteBagTracker
            tracker = MockJuteBagTracker()
            zone_tracker = MockJuteBagTracker() # Reuse for simplicity
            
    yield
    # Clean up on shutdown if needed
    print("Shutting down JuteBagTracker...")
    if multi_cam:
        multi_cam.stop_all()
    tracker = None

from fastapi.staticfiles import StaticFiles

# WebSocket Manager with User Scoping
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {} # userId -> [WebSockets]

    async def connect(self, userId: str, websocket: WebSocket):
        await websocket.accept()
        if userId not in self.active_connections:
            self.active_connections[userId] = []
        self.active_connections[userId].append(websocket)

    def disconnect(self, userId: str, websocket: WebSocket):
        if userId in self.active_connections:
            self.active_connections[userId].remove(websocket)
            if not self.active_connections[userId]:
                del self.active_connections[userId]

    async def broadcast(self, message: dict, userId: str = None):
        if userId:
            # Send to specific user
            if userId in self.active_connections:
                for connection in self.active_connections[userId]:
                    try:
                        await connection.send_json(message)
                    except:
                        pass
        else:
            # Global broadcast (system alerts etc)
            for user_conns in self.active_connections.values():
                for connection in user_conns:
                    try:
                        await connection.send_json(message)
                    except:
                        pass

manager = ConnectionManager()

app = FastAPI(lifespan=lifespan, title="CCTV VisionCount AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        # Send initial state (Optional: reset session count for this user?)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)

# --- GLOBAL STATE ---
tasks = {}
# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION_DIR = os.path.join(BASE_DIR, "detections")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # New upload directory
DATA_DIR = os.path.join(BASE_DIR, "data") # Directory for persistent data

# Physical Camera Control Flag
_camera_active = False

class CameraManager:
    _instance = None
    _cap = None

    @classmethod
    def get_cap(cls):
        if cls._cap is None:
            print("Opening Camera Hardware Singleton...")
            cls._cap = cv2.VideoCapture(0)
        return cls._cap

    @classmethod
    def stop(cls):
        if cls._cap is not None:
            print("Force Releasing Camera Hardware Singleton...")
            cls._cap.release()
            cls._cap = None
        return True

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

def process_video_task(task_id: str, video_path: str, mode: str = "static", user_id: str = "anonymous", depth_override: str = None):
    """
    Background task to process video and update status.
    """
    global tracker, zone_tracker, godown_tracker
    if not tracker or ((mode == "zone" or mode == "conveyor") and not zone_tracker) or (mode == "godown" and not godown_tracker):
        print("Tracker(s) not initialized!")
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized", "user_id": user_id}
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
            asyncio.run(manager.broadcast(data, userId=user_id))
        except:
            pass
        
    try:
        # Save output to detections folder with a clean name
        output_filename = f"detected_{task_id}.mp4"
        output_video_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Run tracking and save video with callback
        # v5: Modular Choice between Tracking types
        if mode == "zone" or mode == "conveyor":
            zone_tracker.reset_state() # v10.6 Fix: Prevent count leakage across videos
            results = zone_tracker.process_video(video_path, output_video_path, on_update=safe_broadcast, mode=mode)
        elif mode == "godown":
            # Godown mode: direction-based counting
            line_pos = 0.5  # Default center; frontend can pass custom
            results = godown_tracker.process_video(video_path, output_video_path, line_position=line_pos, on_update=safe_broadcast)
        elif mode == "volume":
            results = volume_estimator.process_video(video_path, output_video_path, on_update=safe_broadcast, depth_override=depth_override)
        else:
            tracker.reset_state() # v10.6 Fix: Standardize reset for all modes
            results = tracker.process_video(video_path, output_video_path, mode=mode, on_update=safe_broadcast)
        
        # Results now contains the count directly from the tracker
        final_count = results.get("count", 0)
        cumulative_total = results.get("total_count", 0) if (mode == "zone" or mode == "conveyor") else 0
        
        # v8.6 reporting: Use cumulative total for upload status list
        reported_count = cumulative_total if (mode == "zone" or mode == "conveyor") else final_count
        
        # Force a final broadcast of the global total to ensure UI is in sync
        # v13.0 Precision Fix: Broadcast ONLY the current task's count.
        # This prevents the Summation Bug (6 bag bug)
        safe_broadcast({"count": reported_count})
        
        task_data = {
            "status": "completed",
            "count": reported_count,
            "results_count": reported_count,
            "video_url": f"/download/{output_filename}"
        }
        
        # Inject volume estimation extended results if applicable
        if mode == "volume" and results.get("estimation_mode"):
            task_data.update({
                "estimation_mode": True,
                "visible_count": results.get("visible_count", 0),
                "depth_layers": results.get("depth_layers", 0),
                "estimated_total": results.get("estimated_total", 0),
                "depth_override_used": results.get("depth_override_used", False)
            })
            
        tasks[task_id] = task_data
        save_tasks()
        
        # Optional: Clean up input file after processing
        # if os.path.exists(video_path):
        #     os.remove(video_path)
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()

def process_image_task(task_id: str, image_path: str, mode: str = "static", user_id: str = "anonymous", depth_override: str = None):
    """
    Background task to process an image.
    """
    global tracker
    if not tracker:
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized", "user_id": user_id}
        save_tasks()
        return

    print(f"Starting image task {task_id} for {image_path}")
    
    # Callback for real-time updates
    def safe_broadcast(data: dict):
        asyncio.run(manager.broadcast(data, userId=user_id))
    
    try:
        output_filename = f"detected_{task_id}.jpg"
        output_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Run processing with callback
        if mode == "volume":
            results = volume_estimator.process_image(image_path, output_path, on_update=safe_broadcast, depth_override=depth_override)
        else:
            results = tracker.process_image(image_path, output_path, on_update=safe_broadcast)
        
        # Add to task results
        results["video_url"] = f"/download/{output_filename}" # Frontend expects video_url for display
        results["is_image"] = True # Flag for frontend
        results["user_id"] = user_id
        
        # Ensure the frontend knows if we used an override
        if mode == "volume" and results.get("estimation_mode"):
            results.setdefault("depth_override_used", False)
            
        tasks[task_id] = results
        save_tasks()
        
        # CRITICAL FIX: Broadcast the final processed image so it appears in the frontend Live Feed 
        try:
            import base64
            img = cv2.imread(output_path)
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                safe_broadcast({
                    "type": "frame", 
                    "data": frame_data, 
                    "count": results.get("visible_count", results.get("count", 0)), 
                    "results_count": results.get("estimated_total", results.get("count", 0))
                })
        except Exception as e:
            print(f"Failed to broadcast image frame: {e}")

        asyncio.run(manager.broadcast({"count": results.get("count", 0)}, userId=user_id))
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    mode: str = Form("static"),
    user_id: str = Form("anonymous"),
    depth_override: str = Form(None)
):
    """
    Uploads a file (Video or Image) and starts processing.
    """
    # Generate unique ID
    task_id = str(uuid.uuid4())
    filename = file.filename.lower()
    
    # Save file using chunked streaming to avoid loading entire file into RAM
    file_location = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_location, "wb") as file_object:
        while True:
            chunk = await file.read(1024 * 1024)  # Read 1 MB at a time
            if not chunk:
                break
            file_object.write(chunk)
    
    # Determine type
    is_image = filename.endswith(('.jpg', '.jpeg', '.png', '.webp'))
    
    # Validation based on Mode
    if mode == "static" and not is_image:
        return JSONResponse(status_code=400, content={"message": "Static Mode strictly supports IMAGES only (JPG, PNG). Please upload an image."})
    
    if mode == "scanning" and is_image:
        return JSONResponse(status_code=400, content={"message": "Scanning Mode supports VIDEOS only. Please upload a video."})

    if (mode == "zone" or mode == "conveyor") and is_image:
        return JSONResponse(status_code=400, content={"message": "Zone Mode supports VIDEOS only. Please upload a video."})

    if mode == "godown" and is_image:
        return JSONResponse(status_code=400, content={"message": "Godown Mode supports VIDEOS only. Please upload a video."})

    # NOTE: Do NOT reset global tracker here — another user may be processing concurrently.
    # The background task resets the tracker right before it starts processing.
    
    # Initial task status
    tasks[task_id] = {"status": "processing", "progress": 0, "file": file.filename, "mode": mode, "user_id": user_id, "depth_override": depth_override}
    save_tasks()
    
    # Start background processing
    if is_image:
        background_tasks.add_task(process_image_task, task_id, file_location, mode, user_id, depth_override)
    else:
        background_tasks.add_task(process_video_task, task_id, file_location, mode, user_id, depth_override)
    
    return {"task_id": task_id, "message": "Upload accepted and processing started."}

@app.post("/reset")
async def reset_session(user_id: str = "anonymous"):
    """Resets the session count for a specific user only."""
    # NOTE: We do NOT reset the global tracker here — another user may be processing.
    # The tracker resets right before processing starts in the background task.
    
    # Broadcast reset ONLY to the requesting user's WS connections
    await manager.broadcast({"count": 0, "event": "reset"}, userId=user_id)
    return {"message": "Session reset successfully", "count": 0}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

def generate_frames():
    """
    Generator for camera stream using Singleton Manager. 
    """
    global tracker, _camera_active
    if not tracker:
        return

    cap = CameraManager.get_cap()
    
    try:
        while _camera_active:
            success, frame = cap.read()
            if not success:
                # If camera fails during stream, try to reset singleton
                CameraManager.stop()
                break
            
            # Run Live AI Processing
            frame = tracker.process_live_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small sleep to yield control
            import time
            time.sleep(0.01)
    finally:
        # We don't release here anymore, we wait for explicit /camera/off
        print("Stream Generator segment ended.")

@app.get("/stream")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/camera/on")
async def camera_on():
    global _camera_active
    _camera_active = True
    print("UI Requested Camera ON")
    return {"status": "camera_powering_up"}

@app.post("/camera/off")
async def camera_off():
    global _camera_active
    _camera_active = False
    CameraManager.stop() # HARD STOP HARDWARE
    print("UI Requested Camera OFF - HARDWARE KILLED")
    return {"status": "camera_shutting_down"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================
# MULTI-CCTV MODE ENDPOINTS
# ============================================================

@app.post("/multi-cctv/add")
async def multi_cctv_add(label: str = Form("")):
    """Add a new camera cell."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    
    camera_id = str(uuid.uuid4())[:8]
    result = multi_cam.add_camera(camera_id, label=label)
    return result

@app.post("/multi-cctv/remove/{camera_id}")
async def multi_cctv_remove(camera_id: str):
    """Remove a camera cell."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return multi_cam.remove_camera(camera_id)

@app.post("/multi-cctv/upload/{camera_id}")
async def multi_cctv_upload(
    camera_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("anonymous")
):
    """Upload a video to a specific camera cell for processing."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    
    # Save file
    task_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_location, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    
    output_filename = f"multicam_{camera_id}_{task_id}.mp4"
    output_path = os.path.join(DETECTION_DIR, output_filename)
    
    # Callback for broadcasts
    def cam_broadcast(data):
        try:
            asyncio.run(manager.broadcast(data, userId=user_id))
        except:
            pass
    
    # Process in background
    background_tasks.add_task(
        multi_cam.process_camera_video,
        camera_id, file_location, output_path, cam_broadcast
    )
    
    return {"status": "processing", "camera_id": camera_id, "task_id": task_id}

@app.post("/multi-cctv/live/{camera_id}")
async def multi_cctv_live(camera_id: str, source: str = Form("0")):
    """Start live detection on a camera source."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return multi_cam.start_live(camera_id, source)

@app.get("/multi-cctv/stream/{camera_id}")
def multi_cctv_stream(camera_id: str):
    """MJPEG stream for a specific camera."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return StreamingResponse(
        multi_cam.generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/multi-cctv/counts")
def multi_cctv_counts():
    """Get per-camera counts + total."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return multi_cam.get_counts()

@app.post("/multi-cctv/stop")
async def multi_cctv_stop():
    """Stop all camera feeds."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return multi_cam.stop_all()

@app.post("/multi-cctv/stop/{camera_id}")
async def multi_cctv_stop_camera(camera_id: str):
    """Stop a specific camera's live feed."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    return multi_cam.stop_camera(camera_id)

@app.post("/multi-cctv/upload-grid")
async def multi_cctv_upload_grid(
    file: UploadFile = File(...),
    rows: int = Form(2),
    cols: int = Form(2),
    user_id: str = Form("anonymous")
):
    """Upload a single multi-camera grid video, split it, and process each quadrant in PARALLEL."""
    global multi_cam
    if not multi_cam:
        raise HTTPException(status_code=500, detail="Multi-camera manager not initialized")
    
    import threading
    
    # Save uploaded file
    upload_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{upload_id}_{file.filename}")
    with open(file_location, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    
    # Split the video into quadrants
    split_output_dir = os.path.join(UPLOAD_DIR, f"split_{upload_id}")
    try:
        split_results = split_video(file_location, rows, cols, split_output_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to split video: {str(e)}")
    
    # Create camera cells and launch PARALLEL threads for each
    created_cameras = []
    for split in split_results:
        camera_id = str(uuid.uuid4())[:8]
        multi_cam.add_camera(camera_id, label=split["label"])
        
        output_filename = f"multicam_{camera_id}_{upload_id}.mp4"
        output_path = os.path.join(DETECTION_DIR, output_filename)
        
        # Broadcast callback scoped to this user
        def make_broadcast(uid):
            def cam_broadcast(data):
                try:
                    asyncio.run(manager.broadcast(data, userId=uid))
                except:
                    pass
            return cam_broadcast
        
        # Launch each camera in its own thread for TRUE parallel processing
        thread = threading.Thread(
            target=multi_cam.process_camera_video,
            args=(camera_id, split["video_path"], output_path, make_broadcast(user_id)),
            daemon=True
        )
        thread.start()
        
        created_cameras.append({
            "camera_id": camera_id,
            "label": split["label"]
        })
    
    return {
        "status": "processing",
        "cameras": created_cameras,
        "grid": f"{rows}x{cols}",
        "total_cameras": len(created_cameras)
    }


# ============================================================
# GODOWN MODE ENDPOINTS
# ============================================================

@app.get("/godown/status")
def godown_status():
    """Get current godown inventory status."""
    global godown_tracker
    if not godown_tracker:
        raise HTTPException(status_code=500, detail="Godown tracker not initialized")
    return godown_tracker.get_status()

@app.post("/godown/set-baseline")
async def godown_set_baseline(count: int = Form(0)):
    """Manually set the godown inventory baseline."""
    global godown_tracker
    if not godown_tracker:
        raise HTTPException(status_code=500, detail="Godown tracker not initialized")
    return godown_tracker.set_baseline(count)

@app.post("/godown/reset-daily")
async def godown_reset_daily():
    """Reset daily in/out counters."""
    global godown_tracker
    if not godown_tracker:
        raise HTTPException(status_code=500, detail="Godown tracker not initialized")
    return godown_tracker.reset_daily()

# Godown Live Camera State
_godown_live_active = False
_godown_line_pos = 0.5

@app.post("/godown/start-live")
async def godown_start_live(line_position: float = Form(50), user_id: str = Form("anonymous")):
    """Start godown CCTV live monitoring."""
    global godown_tracker, _godown_live_active, _godown_line_pos
    if not godown_tracker:
        raise HTTPException(status_code=500, detail="Godown tracker not initialized")
    
    # Convert line_position from percentage (10-90) to fraction (0.1-0.9)
    _godown_line_pos = line_position / 100.0
    _godown_live_active = True
    godown_tracker.reset_state()
    if hasattr(godown_tracker, '_live_frame_idx'):
        godown_tracker._live_frame_idx = 0
    
    print(f"Godown CCTV Live started. Line position: {_godown_line_pos:.2f}")
    return {"status": "godown_live_started", "line_position": _godown_line_pos}

@app.post("/godown/update-line")
async def godown_update_line(line_position: float = Form(50)):
    """Dynamically update godown counting line position."""
    global _godown_line_pos
    _godown_line_pos = line_position / 100.0
    return {"status": "success", "line_position": _godown_line_pos}

@app.post("/godown/stop-live")
async def godown_stop_live():
    """Stop godown CCTV live monitoring."""
    global _godown_live_active
    _godown_live_active = False
    CameraManager.stop()
    print("Godown CCTV Live stopped")
    return {"status": "godown_live_stopped"}

def generate_godown_frames():
    """Generator for godown CCTV MJPEG stream."""
    global godown_tracker, _godown_live_active, _godown_line_pos
    if not godown_tracker:
        return

    cap = CameraManager.get_cap()
    
    try:
        while _godown_live_active:
            success, frame = cap.read()
            if not success:
                CameraManager.stop()
                break
            
            # Process frame through godown tracker (with person filtering)
            annotated = godown_tracker.process_live_frame(frame, line_position=_godown_line_pos)
            
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            import time
            time.sleep(0.01)
    finally:
        print("Godown stream generator ended.")

@app.get("/godown/stream")
def godown_stream():
    """MJPEG stream for godown live CCTV monitoring."""
    return StreamingResponse(
        generate_godown_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

