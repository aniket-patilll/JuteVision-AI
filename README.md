# CCTV VisionCount AI - Automated Jute Bag Counter

An AI-powered system for automated counting of jute bags using YOLOv8 object detection and tracking.

## 🎯 Features

- **Real-time Jute Bag Detection** - YOLOv8-powered object detection with custom fine-tuned weights
- **Tiled Detection (SAHI-lite)** - Optimized for high-resolution static warehouse pile images
- **Dynamic Analysis Modes** - Specialized logic for **Conveyor**, **Static**, **Scanning**, **Zone**, **Quantity Count Pro Mode**, **Multi-CCTV**, and **Godown** processing
- **Download Sample Feature** - Instant ZIP download of high-quality sample videos and images for each analysis mode
- **High-Density Flow Optimization** - Enhanced deduplication and ID jump protection for rapid product streams
- **Session-Based Isolation** - Full data and state isolation for multiple concurrent users and guest sessions
- **Live CCTV Integration** - Real-time MJPEG camera feed with live ROI occupancy metrics
- **Interactive Analytics Dashboard** - Premium glassmorphism UI with real-time charts (ApexCharts), activity logs, and CSV export
- **WebSocket Infrastructure** - Low-latency, bi-directional communication for instant dashboard updates
- **Internal Tools** - Built-in scripts for automated testing (`test_dummy.jpg`) and AI model training (`train_sacks_model.py`)
- **Session Isolation** - Full data and state isolation for multiple users and guest sessions

## 🏗️ Technology Stack

### Backend
- **Flask** - Lightweight Python web framework
- **Flask-Sock** - WebSocket support for Flask
- **YOLOv8 (ultralytics)** - Object detection and tracking
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing
- **Threading** - Asynchronous AI processing

### Frontend
- **Vite** - Next-generation build tool
- **Vanilla JavaScript** - Lightweight and fast
- **Chart.js** - Interactive data visualization for analytics dashboard
- **CSS3** - Custom design system with modern components

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/saipratyushap/CCTV-VisionCount-AI.git
cd CCTV-VisionCount-AI
```

### 2. Download YOLOv8 Model
Download the YOLOv8 medium model and place it in `backend/models/`:
```bash
# Visit https://github.com/ultralytics/assets/releases
# Download yolov8m.pt
# Move it to backend/models/yolov8m.pt
```

Or use Python:
```bash
cd backend
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
mv yolov8m.pt models/
```

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 4. Frontend Setup
```bash
cd frontend
npm install
```

## 🎮 Usage

### Start Backend Server
```bash
cd backend
.venv/bin/python3 -m app.flask_main
```

The backend will be available at `http://localhost:8000`

### Start Frontend Server
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 📁 Project Structure

```
CCTV_VisionCount_AI/
├── backend/
│   ├── app/
│   │   ├── flask_main.py       # Primary Flask backend server
│   │   ├── main.py             # (Legacy) FastAPI backend server
│   │   ├── tracker.py          # YOLOv8 tracker with high-density logic
│   │   ├── zone_tracker.py     # ROI tracking for Conveyor and custom Zone modes
│   ├── data/                   # NEW: Consolidated storage for all dynamic data
│   │   ├── detections/         # Processed media output (Annotated videos/images)
│   │   ├── uploads/            # Raw temporary uploads (Videos/Images)
│   │   ├── tasks.json          # Persistent record of processing history
│   │   └── godown_inventory.json # Persistent warehouse stock monitoring
│   ├── models/
│   │   ├── yolov8n.pt          # Base YOLOv8 model
│   │   └── sacks_custom.pt     # Fine-tuned weights for jute bags
│   ├── requirements.txt
├── frontend/
│   ├── index.html              # Main Monitoring Dashboard
│   ├── analytics.html          # Glassmorphism Data & Trends Dashboard
│   ├── script.js               # Application logic & ZIP bundling
│   ├── style.css               # Premium design system
│   └── assets/
│       └── samples/            # Pre-loaded sample bundles (.mp4, .jpg)
└── README.md
```

## 🔧 Configuration

### Analysis Modes
The system supports seven distinct analysis modes:

1. **Conveyor Mode** (Optimized for Industrial Belts)
   - High-precision tracking for standard conveyor systems with optimized ROI logic.
2. **Static Mode** (Optimized for Images)
   - Uses **Tiled Detection (SAHI-lite)** to count stationary bags in high-res warehouse stacks.
3. **Scanning Mode** (Optimized for Video)
   - Uses a **Center Scanning Zone** logic for dynamic scenes.
4. **Zone Counting Mode** (Optimized for Custom Areas)
   - Tracks objects crossing defined boundaries in specialized flow environments with custom ROI definitions.
5. **Quantity Count Pro Mode** (Optimized for 3D Estimation)
   - Geometrically predicts the total hidden volume of stacked sacks based on visible 2D dimensions and depth heuristics.
6. **Multi-CCTV Surveillance Mode** (Optimized for Scale)
   - Monitors a dynamic number of camera feeds simultaneously within a grid, supporting both live streams and video uploads.
7. **Godown (Counting & Decounting) Mode** (Optimized for Logistics)
   - Dual-directional tracking with a user-adjustable counting line to monitor stock entering and leaving the facility.

The **Live Feed** toggle on the dashboard allows for real-time monitoring directly from connected CCTV sources.

## 💾 Data Storage & Management

The system uses a hybrid storage approach to ensure performance and reliability:

### Backend (Server-Side)
- **Active Detections**: Annotated videos and images generated by the live application are stored in `backend/data/detections/`. This folder is required for the dashboard to display your results.
- **Original Uploads**: Raw files uploaded by users are temporarily kept in `backend/data/uploads/`.
- **Task History**: Permanent records of processing tasks are saved in `backend/data/tasks.json`.
- **Session Isolation**: Godown inventory for each user is stored separately in `backend/data/godown_inventory_{user_id}.json`.

### Frontend (Client-Side Analytics)
- **Browser LocalStorage**: Primary storage for real-time dashboard persistence:
    - `analyticsData`: **The main database for the Analytics Tab**. Stores up to 50 processed task logs (Time, Filename, Count, Status).
    - `currentTotalBags`: Tracks the cumulative session count across page reloads.
    - `recentUploads`: Manages the history list shown in the dashboard sidebar.
    - `jutevision_guest_id`: Stable unique ID for anonymous users to ensure session isolation.

## 🔄 Real-Time Dashboard Updates

The dashboard maintains high interactivity through three primary mechanisms:

1. **WebSockets (Push)**: A dedicated `ws://` connection enables the backend to push live count updates and processing statuses directly to the UI without page refreshes.
2. **Task Polling (Pull)**: After an upload, the dashboard polls the status of the specific `task_id` until completion.
3. **Session Persistence**: On page load, the frontend synchronizes with `localStorage` to restore previous counts and activity logs immediately.

## 📊 API Endpoints

- `POST /upload` - Upload video/image for processing
- `GET /tasks/{task_id}` - Get processing status
- `GET /stream/{user_id}` - MJPEG live camera stream
- `WS /ws/{user_id}` - WebSocket for real-time updates
- `POST /reset` - Reset current session data
- `GET /godown/status/{user_id}` - Get Godown inventory metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [Flask](https://flask.palletsprojects.com/) for the backend framework
- [Vite](https://vitejs.dev/) for the frontend build tool

## 📧 Contact

For questions or support, please open an issue on GitHub.
