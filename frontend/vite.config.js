import { defineConfig } from 'vite'

export default defineConfig({
    server: {
        proxy: {
            '/upload': 'http://localhost:8000',
            '/tasks': 'http://localhost:8000',
            '/stream': 'http://localhost:8000',
            '/download': 'http://localhost:8000',
            '/ws': {
                target: 'ws://localhost:8000',
                ws: true
            }
        }
    }
})
