# CCTV VisionCount AI - Automated Jute Bag Counter

An AI-powered system for automated counting of jute bags using YOLOv8 object detection and tracking.

## 🎯 Features

- **Real-time Jute Bag Detection** - YOLOv8-powered object detection
- **Tiled Detection (SAHI-lite)** - Accurate counting of small objects in high-res images
- **Automatic Counting** - Tracks unique bags with persistent IDs
- **High-Density Flow Optimization** - Robust deduplication and ID jump protection for rapid product flow
- **Supabase Authentication** - Secure login with Google OAuth and Email verification
- **Video & Image Analysis** - Process warehouse piles or conveyor videos
- **Live CCTV Streaming** - Real-time MJPEG camera feed integration
- **WebSocket Updates** - Instant count updates to the dashboard
- **Analytics Dashboard** - Premium glassmorphism UI with detailed logs and CSV export
- **Modern Web UI** - Fully responsive design with theme-tailored aesthetics

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
│   │   ├── tracker.py      # YOLOv8 tracker with jump protection
│   │   └── utils.py
│   ├── models/
│   │   └── yolov8m.pt      # YOLOv8 model
│   ├── temp_uploads/       # Processed media
│   └── requirements.txt
├── frontend/
│   ├── index.html          # Landing Page
│   ├── dashboard.html      # Main Monitoring Interface
│   ├── analytics.html      # Glassmorphism Data Dashboard
│   ├── login.html / register.html
│   ├── auth.js            # Supabase Integration
│   ├── script.js          # Dashboard Logic & WS
│   ├── style.css          # Design System
│   └── assets/            # Brand Assets
└── README.md
```

## 🔧 Configuration

### Analysis Modes
The system supports three distinct analysis modes:

1. **Static Mode** (Optimized for Images)
   - Uses **Tiled Detection (SAHI-lite)** to count stationary bags in high-res warehouse stacks.
2. **Scanning Mode** (Optimized for Video)
   - Uses a **Center Scanning Zone** logic for dynamic scenes.
3. **Zone Counting Mode** (Optimized for Conveyors)
   - Tracks objects crossing defined boundaries in specialized flow environments.

The **Live Feed** toggle on the dashboard allows for real-time monitoring directly from connected CCTV sources.

## 📊 API Endpoints

- `POST /upload` - Upload video/image for processing
- `GET /tasks/{task_id}` - Get processing status
- `GET /stream` - MJPEG live camera stream
- `WS /ws` - WebSocket for real-time updates
- `POST /reset` - Reset current session data

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
