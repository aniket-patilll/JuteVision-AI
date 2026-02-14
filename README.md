# CCTV VisionCount AI - Automated Jute Bag Counter

An AI-powered system for automated counting of jute bags using YOLOv8 object detection and tracking.

## 🎯 Features

- **Real-time Jute Bag Detection** - YOLOv8-powered object detection
- **Tiled Detection (SAHI-lite)** - Accurate counting of small objects in high-res images
- **Automatic Counting** - Tracks unique bags with persistent IDs
- **Supabase Authentication** - Secure login with Google OAuth support
- **Video & Image Analysis** - Process warehouse piles or conveyor videos
- **Live CCTV Streaming** - Real-time MJPEG camera feed integration
- **WebSocket Updates** - Instant count updates to the dashboard
- **Analytics Dashboard** - History tracking with CSV data export
- **Modern Web UI** - Premium design with responsive dashboard and modal uploads

## 🏗️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **YOLOv8 (ultralytics)** - Object detection and tracking
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing
- **WebSocket** - Real-time communication

### Frontend
- **Vite** - Next-generation build tool
- **Vanilla JavaScript** - Lightweight and fast
- **Supabase** - Authentication & Backend-as-a-Service
- **CSS3** - Modern styling with custom design system

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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

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
│   │   ├── main.py         # FastAPI application
│   │   ├── tracker.py      # YOLOv8 tracker
│   │   ├── mock_tracker.py # Mock for testing
│   │   └── utils.py
│   ├── models/
│   │   └── yolov8m.pt      # YOLOv8 model (download separately)
│   ├── temp_uploads/       # Processed videos
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── vite.config.js
└── README.md
```

## 🔧 Configuration

### Processing Modes
The system supports two distinct analysis modes, selectable via the UI:

1. **Static Mode** (Optimized for Images/Piles)
   - Uses **Tiled Detection (SAHI-lite)** to scan high-resolution warehouse photos.
   - Best for counting stationary bags stacked in large piles.
2. **Scanning Mode** (Optimized for Video/Conveyors)
   - Uses a **Center Scanning Zone** logic.
   - Bags are counted only when they enter the designated zone in the center of the frame.
   - Prevents double-counting in dynamic warehouse scenes.

The **Live Feed** toggle on the dashboard allows for real-time monitoring and counting from connected CCTV sources.


## 📊 API Endpoints

- `POST /upload` - Upload video for processing
- `GET /tasks/{task_id}` - Get processing status
- `GET /stream` - MJPEG live camera stream
- `WS /ws` - WebSocket for real-time updates
- `GET /download/{filename}` - Download processed video

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Vite](https://vitejs.dev/) for the frontend build tool

## 📧 Contact

For questions or support, please open an issue on GitHub.
