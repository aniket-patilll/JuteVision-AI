import os
import uuid
import json
import threading
import base64
import cv2
import shutil
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

# Import local modules
# Import local modules
from .tracker import JuteBagTracker
from .zone_tracker import ModularZoneTracker
from .godown_tracker import GodownTracker
from .volume_estimator import VolumeEstimator
from .multi_camera_tracker import MultiCameraManager
from .video_splitter import split_video
from .camera_manager import CameraManager
from .mock_tracker import MockJuteBagTracker

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- SESSION MANAGEMENT ---
USER_SESSIONS: dict[str, dict] = {}

def get_user_trackers(user_id: str) -> dict:
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = {
            "tracker": None, "zone_tracker": None, "godown_tracker": None,
            "volume_estimator": None, "multi_cam": None,
            "camera_active": False, "godown_live_active": False, "godown_line_pos": 0.5
        }
    return USER_SESSIONS[user_id]

def ensure_tracker(user_id, key):
    session = get_user_trackers(user_id)
    if session.get(key) is None:
        print(f"Lazy-loading {key} for user {user_id}...")
        try:
            if key == "tracker": session["tracker"] = JuteBagTracker()
            elif key == "zone_tracker": session["zone_tracker"] = ModularZoneTracker()
            elif key == "godown_tracker": session["godown_tracker"] = GodownTracker(user_id=user_id)
            elif key == "volume_estimator": session["volume_estimator"] = VolumeEstimator()
            elif key == "multi_cam": session["multi_cam"] = MultiCameraManager()
        except Exception as e:
            print(f"Failed to load {key}: {e}")
            mock = MockJuteBagTracker()
            session[key] = mock
    return session[key]

# --- WEBSOCKET MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list] = {} # userId -> [socks]
    def connect(self, user_id, ws):
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(ws)
    def disconnect(self, user_id, ws):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(ws)
            if not self.active_connections[user_id]:
                self.active_connections.pop(user_id, None)
    def broadcast(self, message, user_id=None):
        data = json.dumps(message)
        if user_id:
            for s in self.active_connections.get(user_id, []):
                try: s.send(data)
                except: pass
        else:
            for user_conns in self.active_connections.values():
                for s in user_conns:
                    try: s.send(data)
                    except: pass

ws_manager = ConnectionManager()

@sock.route('/ws/<user_id>')
def websocket_endpoint(ws, user_id):
    ws_manager.connect(user_id, ws)
    try:
        while True:
            ws.receive()
    except:
        pass
    finally:
        ws_manager.disconnect(user_id, ws)

# --- GLOBAL STATE & DIRECTORIES ---
tasks = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DETECTION_DIR = os.path.join(DATA_DIR, "detections")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

TASK_FILE = os.path.join(DATA_DIR, "tasks.json")
# Removed global SESSION_ID to support unique client sessions

def load_tasks():
    global tasks
    tasks = {}

def save_tasks():
    pass # Disabled persistence

load_tasks()

@app.route('/session/id')
def session_id():
    return jsonify({"session_id": str(uuid.uuid4())})



@app.route('/download/<path:filename>')
def download_file(filename):
    mimetype = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
    return send_from_directory(DETECTION_DIR, filename, mimetype=mimetype)

# --- BACKGROUND TASK LOGIC ---
def run_in_background(target, args=()):
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()

def process_video_task(task_id, video_path, mode, user_id, depth_override):
    if mode in ["zone", "conveyor"]: 
        tracker = ensure_tracker(user_id, "zone_tracker")
    elif mode == "godown": 
        tracker = ensure_tracker(user_id, "godown_tracker")
    elif mode == "volume": 
        tracker = ensure_tracker(user_id, "volume_estimator")
    else: 
        tracker = ensure_tracker(user_id, "tracker")
    
    # Alias for logic below
    zone_tracker = tracker if mode in ["zone", "conveyor"] else None
    godown_tracker = tracker if mode == "godown" else None
    volume_estimator = tracker if mode == "volume" else None
    main_tracker = tracker if mode not in ["zone", "conveyor", "godown", "volume"] else None
    
    def safe_broadcast(data):
        if task_id in tasks:
            if "progress" in data: tasks[task_id]["progress"] = data["progress"]
            if "count" in data: tasks[task_id]["results_count"] = data["count"]
            save_tasks()
        ws_manager.broadcast(data, user_id)

    try:
        output_filename = f"detected_{task_id}.mp4"
        output_path = os.path.join(DETECTION_DIR, output_filename)
        
        if mode in ["zone", "conveyor"]:
            # Explicitly use the ensured tracker to satisfy static analysis
            zt = ensure_tracker(user_id, "zone_tracker")
            if zt:
                zt.reset_state()
                results = zt.process_video(video_path, output_path, on_update=safe_broadcast, mode=mode)
            else:
                raise Exception("Failed to initialize zone tracker")
        elif mode == "godown":
            gt = ensure_tracker(user_id, "godown_tracker")
            if gt:
                results = gt.process_video(video_path, output_path, on_update=safe_broadcast)
            else:
                raise Exception("Failed to initialize godown tracker")
        elif mode == "volume":
            ve = ensure_tracker(user_id, "volume_estimator")
            if ve:
                results = ve.process_video(video_path, output_path, on_update=safe_broadcast, depth_override=depth_override)
            else:
                raise Exception("Failed to initialize volume estimator")
        else:
            t = ensure_tracker(user_id, "tracker")
            if t:
                t.reset_state()
                results = t.process_video(video_path, output_path, mode=mode, on_update=safe_broadcast)
            else:
                raise Exception("Failed to initialize tracker")

        reported_count = results.get("total_count", 0) if mode in ["zone", "conveyor"] else results.get("count", 0)
        ws_manager.broadcast({"count": reported_count}, user_id)
        
        task_data = {"status": "completed", "count": reported_count, "results_count": reported_count, "video_url": f"/download/{output_filename}"}
        if mode == "volume" and results.get("estimation_mode"):
            task_data.update({"estimation_mode": True, "visible_count": results.get("visible_count", 0), "depth_layers": results.get("depth_layers", 0), "estimated_total": results.get("estimated_total", 0)})
        
        tasks[task_id] = task_data
        save_tasks()
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()

def process_image_task(task_id, image_path, mode, user_id, depth_override):
    if mode == "volume":
        estimator = ensure_tracker(user_id, "volume_estimator")
        proc_func = estimator.process_image
    else:
        tracker = ensure_tracker(user_id, "tracker")
        proc_func = tracker.process_image
    def safe_broadcast(data): ws_manager.broadcast(data, user_id)
    try:
        output_filename = f"detected_{task_id}.jpg"
        output_path = os.path.join(DETECTION_DIR, output_filename)
        if mode == "volume":
            results = estimator.process_image(image_path, output_path, on_update=safe_broadcast, depth_override=depth_override)
        else:
            results = tracker.process_image(image_path, output_path, on_update=safe_broadcast)
        results.update({"video_url": f"/download/{output_filename}", "is_image": True, "user_id": user_id})
        
        # Final frame broadcast
        img = cv2.imread(output_path)
        if img is not None:
            _, buf = cv2.imencode('.jpg', img)
            ws_manager.broadcast({"type": "frame", "data": base64.b64encode(buf).decode('utf-8'), "count": results.get("visible_count", results.get("count", 0)), "results_count": results.get("estimated_total", results.get("count", 0))}, user_id)
        
        tasks[task_id] = results
        save_tasks()
        ws_manager.broadcast({"count": results.get("count", 0)}, user_id)
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}
        save_tasks()

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    mode = request.form.get('mode', 'static')
    user_id = request.form.get('user_id', 'anonymous')
    depth_override = request.form.get('depth_override')
    
    task_id = str(uuid.uuid4())
    filename = file.filename.lower()
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    file.save(file_path)
    
    is_image = filename.endswith(('.jpg', '.jpeg', '.png', '.webp'))
    if mode == "static" and not is_image: return jsonify({"message": "Static Mode requires IMAGES"}), 400
    
    tasks[task_id] = {"status": "processing", "progress": 0, "file": file.filename, "mode": mode, "user_id": user_id}
    save_tasks()
    
    if is_image: run_in_background(process_image_task, (task_id, file_path, mode, user_id, depth_override))
    else: run_in_background(process_video_task, (task_id, file_path, mode, user_id, depth_override))
    
    return jsonify({"task_id": task_id, "message": "Upload accepted"})

@app.route('/tasks/<task_id>')
def get_task(task_id):
    task = tasks.get(task_id)
    return jsonify(task) if task else (jsonify({"error": "Not found"}), 404)

@app.route('/reset', methods=['POST'])
def reset_session():
    user_id = request.args.get('user_id', 'anonymous')
    ws_manager.broadcast({"count": 0, "event": "reset"}, user_id)
    return jsonify({"message": "Reset", "count": 0})

# --- STREAMING ---
from typing import Generator, Iterable
def gen_frames(user_id: str) -> Iterable[bytes]:
    session = get_user_trackers(user_id)
    tracker = ensure_tracker(user_id, "tracker")
    cap = CameraManager.get_cap()
    while session.get("camera_active"):
        try:
            # v14.0 Fix: Use synchronized read to prevent crashes during camera shutdown
            success, frame = CameraManager.read_frame()
            if not success or frame is None: break
            
            frame = tracker.process_live_frame(frame)
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
            break
    print("Stream ended.")

@app.route('/stream/<user_id>')
def stream(user_id):
    return Response(gen_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/on', methods=['POST'])
def camera_on():
    user_id = request.form.get('user_id', 'anonymous')
    get_user_trackers(user_id)["camera_active"] = True
    return jsonify({"status": "on"})

@app.route('/camera/off', methods=['POST'])
def camera_off():
    user_id = request.form.get('user_id', 'anonymous')
    get_user_trackers(user_id)["camera_active"] = False
    CameraManager.stop()
    return jsonify({"status": "off"})

# --- MULTI-CCTV ---
@app.route('/multi-cctv/add', methods=['POST'])
def multi_cctv_add():
    user_id = request.form.get('user_id', 'anonymous')
    label = request.form.get('label', '')
    multi_cam = ensure_tracker(user_id, "multi_cam")
    cam_id = str(uuid.uuid4())[:8] # type: ignore
    return jsonify(multi_cam.add_camera(cam_id, label=label))

@app.route('/multi-cctv/remove/<user_id>/<camera_id>', methods=['POST'])
def multi_cctv_remove(user_id, camera_id):
    return jsonify(ensure_tracker(user_id, "multi_cam").remove_camera(camera_id))

@app.route('/multi-cctv/upload/<camera_id>', methods=['POST'])
def multi_cctv_upload(camera_id):
    user_id = request.form.get('user_id', 'anonymous')
    file = request.files['file']
    task_id = str(uuid.uuid4())
    in_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    file.save(in_path)
    out_path = os.path.join(DETECTION_DIR, f"multicam_{camera_id}_{task_id}.mp4")
    run_in_background(ensure_tracker(user_id, "multi_cam").process_camera_video, (camera_id, in_path, out_path, lambda d: ws_manager.broadcast(d, user_id)))
    return jsonify({"status": "processing", "camera_id": camera_id})

@app.route('/multi-cctv/live/<user_id>/<camera_id>', methods=['POST'])
def multi_cctv_live(user_id, camera_id):
    source = request.form.get('source', '0')
    return jsonify(ensure_tracker(user_id, "multi_cam").start_live(camera_id, source))

@app.route('/multi-cctv/stream/<user_id>/<camera_id>')
def multi_cctv_stream(user_id, camera_id):
    return Response(ensure_tracker(user_id, "multi_cam").generate_mjpeg(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/multi-cctv/counts/<user_id>')
def multi_cctv_counts(user_id):
    return jsonify(ensure_tracker(user_id, "multi_cam").get_counts())

@app.route('/multi-cctv/stop', methods=['POST'])
def multi_cctv_stop_all():
    user_id = request.form.get('user_id', 'anonymous')
    return jsonify(ensure_tracker(user_id, "multi_cam").stop_all())

# --- GODOWN ---
@app.route('/godown/status/<user_id>')
def godown_status(user_id):
    return jsonify(ensure_tracker(user_id, "godown_tracker").get_status())

@app.route('/godown/set-baseline', methods=['POST'])
def godown_baseline():
    user_id = request.form.get('user_id', 'anonymous')
    count = int(request.form.get('count', 0))
    return jsonify(ensure_tracker(user_id, "godown_tracker").set_baseline(count))

@app.route('/godown/reset-daily', methods=['POST'])
def godown_reset():
    user_id = request.form.get('user_id', 'anonymous')
    return jsonify(ensure_tracker(user_id, "godown_tracker").reset_daily())

@app.route('/godown/start-live', methods=['POST'])
def godown_start_live():
    user_id = request.form.get('user_id', 'anonymous')
    line_pos = float(request.form.get('line_position', 50)) / 100.0
    s = get_user_trackers(user_id)
    s["godown_line_pos"], s["godown_live_active"] = line_pos, True
    ensure_tracker(user_id, "godown_tracker").reset_state()
    return jsonify({"status": "started"})

def gen_godown(user_id):
    s = get_user_trackers(user_id)
    cap = CameraManager.get_cap()
    while s.get("godown_live_active"):
        try:
            # v14.0 Fix: Use synchronized read to prevent crashes during camera shutdown
            success, frame = CameraManager.read_frame()
            if not success or frame is None: break

            ann = ensure_tracker(user_id, "godown_tracker").process_live_frame(frame, line_position=s["godown_line_pos"])
            _, buf = cv2.imencode('.jpg', ann)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Godown Stream error: {e}")
            break

@app.route('/godown/stream/<user_id>')
def godown_stream(user_id):
    return Response(gen_godown(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
