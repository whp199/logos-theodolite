import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // the atlas's page carries three.js (about 600 kB), loaded only when that page opens
  build: { chunkSizeWarningLimit: 700 },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
      },
    },
  },
});
