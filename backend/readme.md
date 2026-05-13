# Backend — MediaMTX stream control

FastAPI service that starts and stops publishers to **[MediaMTX](https://github.com/bluenviron/mediamtx)** over RTSP: either **FFmpeg** (`lavfi` synthetic video) or the CUDA binary **`resgrid_plotter`** (synthetic I/Q grids → NVENC). The browser UI talks to **MediaMTX** for HLS/WebRTC; this API controls **which** publisher is active and exposes status, IQ controls, and optional **JPEG frame capture**.

OpenAPI title in code: **MediaMTX stream control**.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, routes, subprocess orchestration, shared-memory IQ controls |
| `app/log_config.py` | Logging setup (console + optional `LOG_FILE`) |
| `scripts/resource_grid_generator.py` | Optional Python producer: full-grid I/Q and/or slot messages into FIFOs |
| `resouce-grid-plotter/` | CUDA/C++ sources for **`resgrid_plotter`** (built in Docker; see below) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Multi-stage image: CUDA build of `resgrid_plotter` + FFmpeg-NVIDIA runtime + FastAPI |

---

## Run locally (Python only)

Requires **Python 3** with a venv or global install.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
```

- **Interactive docs:** [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs) (Swagger), [http://127.0.0.1:8081/redoc](http://127.0.0.1:8081/redoc) (ReDoc), [http://127.0.0.1:8081/openapi.json](http://127.0.0.1:8081/openapi.json).
- **IQ / CUDA paths** expect the **`resgrid_plotter`** binary on `PATH` (see `CUDA_PUBLISHER_BIN`). Without a local CUDA build, use **Docker** (below) or restrict testing to **`SYNTHETIC_VIDEO`** (FFmpeg) if your machine has FFmpeg with the codecs you need.

---

## Run with Docker

From the **repository root** (compose files expect `backend/` in context):

```bash
docker compose build backend   # or your compose service name
docker compose up backend
```

The backend image default command is:

`uvicorn app.main:app --host 0.0.0.0 --port 8081`

The Dockerfile compiles **`resgrid_plotter`** in a CUDA devel stage and copies the binary plus `libcudart` into the runtime image based on **`jrottenberg/ffmpeg`** with NVIDIA support.

---

## HTTP API (summary)

| Area | Endpoints (representative) |
|------|----------------------------|
| Health | `GET /health` |
| Stream mode | `GET` / `POST /api/stream/mode` |
| Publisher | `POST /api/stream/start`, `POST /api/stream/stop` |
| CUDA plotter only | `POST /api/stream/plotter/start`, `POST /api/stream/plotter/stop` |
| Status | `GET /api/stream/status` (encoder stats, IQ rates, optional capturer fields) |
| Playback hints | `GET /api/stream/playback` (WHEP / HLS URLs for the UI) |
| IQ generator (Python FIFO) | `POST /api/stream/generator/start` \| `stop` (aliases under `/api/stream/iq/generator/…`) |
| IQ FPS / pattern | `GET`/`PUT /api/stream/iq/generator-fps`, `GET /api/stream/iq/patterns`, `PUT /api/stream/iq/pattern` |
| Channel legend | `GET /api/stream/iq/channel-legend` |
| Raster → JPEG | `POST /api/stream/plotter/capturer`, `POST /api/stream/plotter/capturer/stop` |

Full request/response schemas and descriptions live in **OpenAPI** (`/docs`).

---

## Environment variables

Behavior is driven by many env vars (MediaMTX host/URL, `STREAM_MODE`, CUDA IQ dimensions, FIFO paths, NVENC preset, CORS, logging, capturer directories, etc.). Authoritative defaults and logic are in **`app/main.py`** (top of file).

**High-signal examples:**

- **`MTX_RTSP_URL`** — RTSP publish URL (default `rtsp://mediamtx:8554/live`).
- **`STREAM_MODE`** / mode API — `SYNTHETIC_VIDEO`, `SYNTHETIC_IQ`, or `SYNTHETIC_CHANNEL_BASED` (slot-message path).
- **`AUTO_START`** — `1` to start the publisher on boot (compose often sets this for dev).
- **`CUDA_*`** — plotter cadence, PRB/symbol counts, FIFO toggles, optional Python generator autostart.
- **`CORS_ORIGINS`**, **`CORS_ORIGIN_REGEX`** — browser access to the API from the Vite dev server or deployed UI.
- **`LOG_FILE`**, **`LOG_LEVEL`** — file logging under `logs/` by default.

For stack-level explanation and diagrams, see **[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)**. For the CUDA plotter pipeline, see **[`../docs/PLOTTER_PIPELINE.md`](../docs/PLOTTER_PIPELINE.md)**. For slot FIFO wire format, see **[`../docs/CHANNEL_BASED_INTERFACE.md`](../docs/CHANNEL_BASED_INTERFACE.md)**.
