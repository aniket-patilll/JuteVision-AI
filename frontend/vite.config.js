import { defineConfig } from 'vite';
import { BACKEND_URL, BACKEND_WS_URL } from './config.js';

export default defineConfig({
    server: {
        proxy: {
            '/upload': BACKEND_URL,
            '/tasks': BACKEND_URL,
            '/stream': BACKEND_URL,
            '/download': BACKEND_URL,
            '/reset': BACKEND_URL,
            '/static': BACKEND_URL, // Common path for backend static files
            '/ws': {
                target: BACKEND_WS_URL,
                ws: true
            }
        }
    }
})
