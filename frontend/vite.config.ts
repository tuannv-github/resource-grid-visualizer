import { defineConfig, loadEnv } from "vite";

/** Origins allowed to load Vite dev modules (default Vite CORS is localhost-only, which breaks LAN IP access). */
const privateDevOrigin =
  /^https?:\/\/(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$/;

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Docker Compose sets this on the container; loadEnv() only merges .env* files, so read process.env too.
  const proxyTarget = (process.env.VITE_DEV_PROXY_TARGET || env.VITE_DEV_PROXY_TARGET || "").trim();
  const proxy = proxyTarget
    ? {
        "/api": { target: proxyTarget, changeOrigin: true },
        "/docs": { target: proxyTarget, changeOrigin: true },
        "/openapi.json": { target: proxyTarget, changeOrigin: true },
        "/redoc": { target: proxyTarget, changeOrigin: true },
      }
    : undefined;

  return {
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      cors: { origin: privateDevOrigin },
      ...(proxy ? { proxy } : {}),
    },
  };
});
