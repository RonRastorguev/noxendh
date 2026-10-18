import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the FastAPI server runs on :8000; Vite proxies its routes so the app uses relative URLs in dev and prod.
const api = 'http://localhost:8000'
const proxied = ['/status', '/baseline', '/scans', '/events', '/scout', '/map', '/files']
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries([
      ...proxied.map((p) => [p, { target: api, changeOrigin: true }]),
      ['/live', { target: api.replace('http', 'ws'), ws: true }],
    ]),
  },
})
