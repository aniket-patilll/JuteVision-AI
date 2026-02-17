// Centralized configuration file for the frontend

// Backend API URL (Source of Truth for Vite Proxy)
export const BACKEND_URL = 'http://localhost:8000';
export const BACKEND_WS_URL = 'ws://localhost:8000';

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
    WS: '/ws'
};

// Supabase Configuration
export const SUPABASE_CONFIG = {
    URL: 'https://nrlogpxkdrflsiukhqcb.supabase.co',
    KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ybG9ncHhrZHJmbHNpdWtocWNiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkxMDk5NTcsImV4cCI6MjA4NDY4NTk1N30.QC6dAmVK6fKainhAs5lltwvDLPJkWg72Wfs4w_nfMRg'
};

// Construct full URLs
export const getApiUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

export const getWsUrl = (endpoint) => {
    // In browser, connect to current host (Vite) which proxies to backend
    if (typeof window !== 'undefined') {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}${endpoint}`;
    }
    return `${BACKEND_WS_URL}${endpoint}`; // Fallback for non-browser environments
};
