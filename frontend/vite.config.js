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
            '/camera': BACKEND_URL, // Proxy camera control endpoints
            '/multi-cctv': BACKEND_URL, // Multi-CCTV mode endpoints
            '/godown': BACKEND_URL, // Godown mode endpoints
            '/ws': {
                target: BACKEND_WS_URL,
                ws: true,
                configure: (proxy) => {
                    // setImmediate: runs AFTER Vite adds its own error listener,
                    // so we can replace it with our silent handler.
                    setImmediate(() => {
                        proxy.removeAllListeners('error');
                        proxy.on('error', (err) => {
                            // Suppress expected disconnect noise from uvicorn --reload
                            if (err.code === 'ECONNRESET' || err.message?.includes('ended by the other party')) return;
                            console.error('[ws proxy]', err.message);
                        });
                    });
                }
            }
        }
    }
})
