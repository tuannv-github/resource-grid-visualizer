# Resource grid visualizer

End-to-end demo stack: a **Vite + TypeScript** UI watches a live stream from **[MediaMTX](https://github.com/bluenviron/mediamtx)** while a **FastAPI** backend starts either **FFmpeg** (synthetic video) or a **CUDA** pipeline (**`resgrid_plotter`**: resource-grid magnitude / channel tint → **NVENC** → RTSP). Optional **Python** producers feed I/Q or slot messages through FIFOs for integrator-style workflows.

---

## Quick start (Docker)

Requires **[Docker](https://docs.docker.com/get-docker/)** with **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** if you use the default GPU-backed backend (see `compose/docker-compose.yml` header for CPU-only notes).

```bash
git clone <repository-url>
cd resource-grid-visualizer
docker compose -f compose/docker-compose.yml watch
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)** (or `http://<your-lan-ip>:5173`). The UI proxies **`/api`**, **`/docs`**, and OpenAPI routes to the backend so you can use **one origin** on port **5173**.

**Typical ports**

| Port | Service |
|------|---------|
| **5173** | Vite frontend |
| **8081** | FastAPI (direct; also reachable via UI proxy in default compose) |
| **8888** | MediaMTX HTTP (LL-HLS) |
| **8889** / **8189** | WebRTC / WHEP + ICE (if you use WebRTC playback) |
| **8554** | RTSP ingest (publisher → MediaMTX) |

For **WebRTC** from another machine, set **`MTX_WEBRTC_ADDITIONAL_HOSTS`** to this host’s LAN IP (see comments in `compose/docker-compose.yml`).

**Variants:** **`compose/docker-compose.dlul.yaml`** (and related `compose/docker-compose.dlul*.yaml`) run **two** backends (**downlink / uplink**) with separate MediaMTX paths; other compose files under **`compose/`** tune bandwidth or encode policy. Read the header comments in each file before changing profiles.

---

## Repository layout

| Area | Path | Notes |
|------|------|--------|
| Backend | [`backend/`](backend/) | FastAPI, `generator.py`, CUDA sources under `backend/plotter/` |
| Frontend | [`frontend/`](frontend/) | Vite app, `ResgridPanel`, hls.js + MediaMTX WHEP helper |
| MediaMTX config | [`mediamtx/`](mediamtx/) | `mediamtx.yml` |
| Compose stacks | [`compose/`](compose/) | `docker-compose*.yml` / `*.yaml` (default stack, DL/UL, PRB / encode variants) |
| Docs | [`docs/`](docs/) | Architecture, plotter pipeline, channel / slot FIFO interface |

Component-level instructions: **[`backend/readme.md`](backend/readme.md)** and **[`frontend/readme.md`](frontend/readme.md)**.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Stack diagram, generation modes, ports, compose behavior, configuration overview |
| [`docs/PLOTTER_PIPELINE.md`](docs/PLOTTER_PIPELINE.md) | **`resgrid_plotter`** assembly → raster → encode |
| [`docs/CHANNEL_BASED_INTERFACE.md`](docs/CHANNEL_BASED_INTERFACE.md) | Slot FIFO wire format, PHY ids, dual DL/UL integrator notes |

---

## Development without Docker

1. Run **MediaMTX** with a config compatible with this repo (or reuse `mediamtx/mediamtx.yml` locally).
2. **Backend:** follow [`backend/readme.md`](backend/readme.md) — local Python needs **`resgrid_plotter`** on `PATH` for IQ modes unless you skip CUDA.
3. **Frontend:** follow [`frontend/readme.md`](frontend/readme.md) — set **`VITE_API_BASE`** (or relative URL + dev proxy) to your API origin.

---

## Naming note

CUDA sources for **`resgrid_plotter`** live in **`backend/plotter/`**; the built binary on `PATH` is **`resgrid_plotter`**.
