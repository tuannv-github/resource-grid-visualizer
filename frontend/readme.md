# Frontend — resource grid visualizer

**Vite + TypeScript** single-page app: one or two **`ResgridPanel`** columns that call the FastAPI **backend** for stream control and play **MediaMTX** output via **hls.js** (LL-HLS) or **WebRTC (WHEP)** using MediaMTX’s browser helper (`src/mediamtx-reader.js`).

Video and HLS/WebRTC traffic go **directly** to MediaMTX from the browser; only **REST** (`/api/...`) hits the backend (optionally through the Vite dev **proxy**).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/app/boot.ts` | Resolves panel layout, mounts **`ResgridPanel`** instances, `beforeunload` cleanup |
| `src/panel/ResgridPanel.ts` | Per-column UI: playback, status polling, mode radios, IQ controls, capturer actions |
| `src/api/resgridApi.ts` | Typed `fetch()` wrappers for stream status, generator FPS, start/stop, capturer, generator |
| `src/playback/mediaPlayback.ts` | HLS vs WHEP wiring, playback URLs from **`GET /api/plotter/playback`** |
| `src/config/panelLayout.ts` | **`VITE_*`** → single vs dual API bases, loopback→LAN host rewrite |
| `src/config/playbackMode.ts` | **`VITE_PLAYBACK_MODE`** + **`?playback=hls`** \| **`webrtc`** |
| `src/config/debug.ts` | **`?debug=1`** / **`VITE_DEBUG=1`** → `[resgrid]` console logs |
| `src/domain/` | Publisher **mode** and status types shared with API shapes |
| `vite.config.ts` | Dev server **:5173**, LAN-friendly CORS, optional **`VITE_DEV_PROXY_TARGET`** proxy |

---

## Prerequisites

- **Node.js** (Docker image uses **22**; local **18+** is typically fine)
- Running **backend** + **MediaMTX** (see repo root compose or [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md))

---

## Local development

```bash
cd frontend
npm ci
npm run dev
```

- Dev server: **[http://127.0.0.1:5173](http://127.0.0.1:5173)** (`strictPort: true`).
- Point the UI at the API: set **`VITE_API_BASE`** (e.g. `http://127.0.0.1:8081`) or use **`VITE_API_RELATIVE=1`** with **`VITE_DEV_PROXY_TARGET=http://127.0.0.1:8081`** in `.env.development.local` so `/api` is proxied (same pattern as Docker Compose).

**Build / preview:**

```bash
npm run build    # tsc && vite build → dist/
npm run preview  # serve production build
```

---

## Docker (dev container)

`Dockerfile.dev` runs **`docker-entrypoint-dev.sh`**: `npm ci` when `package-lock.json` changes, then **`npm run dev -- --host 0.0.0.0 --port 5173`**. Used by repo **`compose/docker-compose*.yml`** services.

---

## Client environment (`VITE_*`)

These are baked in at **build time** (Vite). Restart dev server after changing them.

| Variable | Role |
|----------|------|
| **`VITE_API_RELATIVE`** | When truthy (`1`, `true`, …), API base is **same-origin** (empty string) — use with **`VITE_DEV_PROXY_TARGET`** so the browser only talks to **:5173**. |
| **`VITE_DEV_PROXY_TARGET`** | Vite dev **only**: proxy `/api`, `/docs`, `/openapi.json`, `/redoc` to this URL (e.g. `http://backend:8081`). |
| **`VITE_API_BASE`** | Absolute backend origin when not using relative + proxy. If unset, client defaults to **`http://<page-host>:8081`**. |
| **`VITE_API_BASE_DL`** / **`VITE_API_BASE_UL`** | DL/UL stack: downlink vs uplink FastAPI URLs (with **`VITE_GUI_MODE=dlul`**). |
| **`VITE_API_BASE_RX`** / **`VITE_API_BASE_NAI`** | RX/NAI stack: raw RX vs NAI FastAPI URLs (with **`VITE_GUI_MODE=rxnai`**). |
| **`VITE_GUI_MODE`** | **`combine`** (single column), **`dlul`**, **`rxnai`**, or **`auto`** (infer from API bases). Compose sets this per stack. |
| **`VITE_GUI_COMBINE`** / **`VITE_GUI_DLUL`** / **`VITE_GUI_ULNAI`** | Deprecated; use **`VITE_GUI_MODE`** instead. |
| **`VITE_PLAYBACK_MODE`** | Default **`webrtc`** (WHEP). Set **`hls`** for LL-HLS. Per load: **`?playback=hls`** or **`?playback=webrtc`**. |
| **`VITE_HLS_URL`** | Optional fixed HLS base when playback resolution needs a fallback (see `mediaPlayback.ts`). |
| **`VITE_DEBUG`** | Set **`1`** for verbose **`[resgrid]`** logs; or open the app with **`?debug=1`**. |

If **`VITE_API_*`** uses **`localhost`** / **`127.0.0.1`** but you open the UI from a **LAN** hostname, the client **rewrites** the API host to **`window.location.hostname`** (`panelLayout.ts`) so `fetch` and Swagger links stay reachable.

---

## Related documentation

- **[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)** — full stack, ports, compose profiles, DL/UL behavior.
- **[../backend/readme.md](../backend/readme.md)** — API and backend env vars.
