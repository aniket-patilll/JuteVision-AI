from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
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
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# In-memory task store
tasks = {}
TEMP_DIR = "backend/temp_uploads" # Use the correct path relative to root if running from root
os.makedirs(TEMP_DIR, exist_ok=True)

# Mount static files for video download
app.mount("/download", StaticFiles(directory=TEMP_DIR), name="download")

def process_video_task(task_id: str, video_path: str):
    """
    Background task to process video and update status.
    """
    global tracker
    if not tracker:
        print("Tracker not initialized!")
        tasks[task_id] = {"status": "failed", "error": "Tracker not initialized"}
        return

    print(f"Starting task {task_id} for {video_path}")
    
    # Callback for real-time updates
    def on_update(data: dict):
        import asyncio
        # We need to run the async broadcast from this sync callback
        # Ideally, we should use a proper async loop or queue, but for simplicity:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.broadcast(data))
            loop.close()
        except Exception as e:
            # Fallback for running inside an existing loop (which is complex in sync method)
            # A better way for production is using a Queue or run_in_executor
            print(f"WS Broadcast error: {e}")

    # Simplified Async Wrapper for the callback if running in thread
    # Since background tasks run in a threadpool, we can use asyncio.run or new loop
    def safe_broadcast(data: dict):
        asyncio.run(manager.broadcast(data))

    try:
        output_video_path = f"{TEMP_DIR}/{task_id}_out.mp4"
        
        # Run tracking and save video with callback
        # Note: 'process_video' is blocking, so we pass a lambda that runs the async broadcast
        # Default to "static" mode for warehouse videos (counts all unique bags)
        # Use "conveyor" mode for moving belt scenarios
        results = tracker.process_video(video_path, output_video_path, mode="static", on_update=safe_broadcast)
        
        tasks[task_id] = {
            "status": "completed",
            "count": tracker.total_count,
            "results_count": len(results),
            "video_url": f"/download/{task_id}_out.mp4"
        }
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{task_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(process_video_task, task_id, file_path)
    
    return {"task_id": task_id, "message": "Video uploaded and processing started."}

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
        
        # DUMMY: Just annotate with current count
        cv2.putText(frame, f"Live Stream - Count: {tracker.total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
