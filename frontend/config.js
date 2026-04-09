// Centralized configuration file for the frontend

// Backend API URL (Source of Truth for Vite Proxy)
export const BACKEND_URL = 'http://127.0.0.1:8000';
export const BACKEND_WS_URL = 'ws://127.0.0.1:8000';

// API Base URL for Frontend (Empty to use relative paths/proxy)
export const API_BASE_URL = '';

// WebSocket URL for Frontend 
// (Connects to window.location.host, which Vite proxies to backend)
export const WS_BASE_URL = ''; // Not used directly, using helper below

// API Endpoints
export const ENDPOINTS = {
    UPLOAD: '/upload',
    TASKS: '/tasks', // Append /:taskId
    STREAM: '/stream',
    RESET: '/reset',
    WS: '/ws',
    CAMERA_ON: '/camera/on',
    CAMERA_OFF: '/camera/off',
    // Multi-CCTV
    MULTI_CCTV_ADD: '/multi-cctv/add',
    MULTI_CCTV_UPLOAD: '/multi-cctv/upload',   // Append /:cameraId
    MULTI_CCTV_LIVE: '/multi-cctv/live',       // Append /:cameraId
    MULTI_CCTV_REMOVE: '/multi-cctv/remove',   // Append /:cameraId
    MULTI_CCTV_STOP: '/multi-cctv/stop',       // Append /:cameraId (optional)
    MULTI_CCTV_STREAM: '/multi-cctv/stream',   // Append /:cameraId
    MULTI_CCTV_COUNTS: '/multi-cctv/counts',
    MULTI_CCTV_UPLOAD_GRID: '/multi-cctv/upload-grid',
    // Godown
    GODOWN_STATUS: '/godown/status',
    GODOWN_BASELINE: '/godown/set-baseline',
    GODOWN_RESET_DAILY: '/godown/reset-daily',
    GODOWN_START_LIVE: '/godown/start-live',
    GODOWN_UPDATE_LINE: '/godown/update-line',
    GODOWN_STREAM: '/godown/stream',
    SESSION_ID: '/session/id',
};

// Construcut full URLs
export const getApiUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

export const getWsUrl = (endpoint) => {
    // In browser, connect to current host (Vite) which proxies to backend
    if (typeof window !== 'undefined') {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}${endpoint}`;
    }
    return `${BACKEND_WS_URL}${endpoint}`; // Fallback for non-browser environments
};
