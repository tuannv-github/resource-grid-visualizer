# Backend — MediaMTX stream control

FastAPI service that starts and stops publishers to **[MediaMTX](https://github.com/bluenviron/mediamtx)** over RTSP: either **FFmpeg** (`lavfi` synthetic video) or the CUDA binary **`resgrid_plotter`** (synthetic I/Q grids → NVENC). The browser UI talks to **MediaMTX** for HLS/WebRTC; this API controls **which** publisher is active and exposes status, IQ controls, and optional **JPEG frame capture**.

OpenAPI title in code: **MediaMTX stream control**.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, routes, subprocess orchestration, shared-memory IQ controls |
| `app/log_config.py` | Logging setup (console + optional `LOG_FILE`) |
| `scripts/generator.py` | Optional Python producer: resource-grid I/Q, FAPI/slot messages, and related FIFO feeds |
| `plotter/` | CUDA/C++ sources for **`resgrid_plotter`** (built in Docker; see below) |
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
- **IQ / CUDA paths** expect the **`resgrid_plotter`** binary on `PATH` (see `CUDA_PUBLISHER_BIN`). Without a local CUDA build, use **Docker** (below) or restrict testing to **`SYNTHETIC_IMAGES`** (FFmpeg) if your machine has FFmpeg with the codecs you need.

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
| Session stop | `POST /api/stop` (FFmpeg + CUDA plotter) |
| Generator (**generator.py** / FFmpeg) | `GET /api/generator/status`; `POST /api/generator/start` (JSON **`mode`** or **`iq_scope`**); `POST /api/generator/stop` |
| CUDA plotter | `POST /api/plotter/start`, `POST /api/plotter/stop`, `GET /api/plotter/status` |
| Playback hints | `GET /api/plotter/playback` (WHEP / HLS URLs for the UI) |
| IQ FPS / pattern | `GET`/`PUT /api/generator/fps`, `GET /api/generator/patterns`, `PUT /api/generator/pattern` |
| Channel legend | `GET /api/generator/channel-legend` |
| Raster → JPEG | `POST /api/plotter/capturer`, `POST /api/plotter/capturer/stop` |

Full request/response schemas and descriptions live in **OpenAPI** (`/docs`).

---

## Environment variables

Behavior is driven by many env vars (MediaMTX host/URL, **`GENERATION_MODE`**, CUDA IQ dimensions, FIFO paths, NVENC preset, CORS, logging, capturer directories, etc.). Authoritative defaults and logic are in **`app/main.py`** (top of file).

**High-signal examples:**

- **`MTX_RTSP_URL`** — RTSP publish URL (default `rtsp://mediamtx:8554/live`).
- **`GENERATION_MODE`** — `SYNTHETIC_IMAGES`, `SYNTHETIC_RESOURCE_GRID`, or `SYNTHETIC_FAPI` (process default at startup); **`POST /api/generator/start`** JSON **`mode`** selects the synthetic **generator.py** feed when you use that branch.
- **`AUTO_START`** — `1` to start the publisher on boot; **`compose/`** stacks default **`AUTO_START=0`** (idle until Start / **POST /api/generator/start** or **POST /api/plotter/start**).
- **`CUDA_*`** — plotter cadence, PRB/symbol counts, FIFO toggles, optional Python generator autostart.
- **`CORS_ORIGINS`**, **`CORS_ORIGIN_REGEX`** — browser access to the API from the Vite dev server or deployed UI.
- **`LOG_FILE`**, **`LOG_LEVEL`** — file logging under `logs/` by default.

For stack-level explanation and diagrams, see **[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)**. For the CUDA plotter pipeline, see **[`../docs/PLOTTER_PIPELINE.md`](../docs/PLOTTER_PIPELINE.md)**. For slot FIFO wire format, see **[`../docs/CHANNEL_BASED_INTERFACE.md`](../docs/CHANNEL_BASED_INTERFACE.md)**.
