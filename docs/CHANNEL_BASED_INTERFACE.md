# Channel-based resource grid — interface guide

This document is for **integrators** who want to drive **`resgrid_plotter`** with **logical channel** (PHY-style) metadata: colored regions on the live heatmap, on-screen labels, and optional **slot allocation** records. It complements the stack overview in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. Concepts

| Term | Meaning |
|------|---------|
| **Generation mode** | **`SYNTHETIC_RESOURCE_GRID`** — **full-grid** path: built-in **`--data-mode 1`** by default, or **`--data-mode 2`** + **`--fifo-resource-grid`** when **`CUDA_IQ_USE_FIFO=1`** (each FIFO message = one **complete** grid). **`SYNTHETIC_FAPI`** (historic **`SYNTHETIC_CHANNEL_BASED`**) — **slot messages** / channel-based (`--data-mode 3`): each slot FIFO message is a **slot allocation** (a **part** of the grid for one slot) plus PHY metadata; plotter **assembles** full grids. Channel-based mode always opens **slot FIFO** (`--fifo-fapi`). PHY channel ids **`1…8`** come only from slot messages (OAI **`resgrid_plotter_slot_fifo`** emits **7**/**8** when CSI-RS/SRS PDUs are present). Built-in synthetic I/Q never paints `chan`. Configure with **`GENERATION_MODE`** in compose (default **`synthetic_channel_based`** → **`SYNTHETIC_FAPI`**). Legacy **`CHANNEL_BASED_SYNTHETIC_GENERATION`** aliases channel-based mode. |
| **Logical channel id** | Per-cell byte **`0…255`** in the plotter’s `chan` plane. **`0`** = no channel tint → **Viridis** colormap from magnitude only. **`1…8`** = fixed **PHY-style** RGB tints (see §4). **`9…255`** fall through to **neutral gray** in CUDA until explicitly mapped. |
| **`slot_id`** | Unsigned integer grouping **multiple FIFO messages** into one **full-grid** frame. When `slot_id` **increases**, the previous slot is considered **complete** and is queued for encode (see §6). |

**Frame rates (standalone `resgrid_plotter`):** **`--fps`** (default **200**) is the pipeline / internal producer nominal rate. **`--encoder-fps`** (default **24**) is the **nominal** NVENC timeline (Hz). If **`--fps`** but no **`--encoder-fps`**, both match (single-knob). The backend maps **`CUDA_IQ_GEN_FPS`** → **`--fps`** and **`CUDA_IQ_FPS`** → **`--encoder-fps`**. **`--encode-policy`** (env **`CUDA_ENCODE_POLICY`**) selects how faster CUDA output maps to that rate: **`last`** (default, latest RGBA per tick), **`mean`** (per-pixel **average color in linear RGB**: each input frame’s R/G/B are converted sRGB → linear, averaged, then linear → sRGB; alpha is the rounded mean of input alphas), **`ffmpeg`** (write every rendered frame at **`--fps`** to rawvideo; FFmpeg **`fps=`** filter → **`--encoder-fps`** before NVENC; idle gaps still duplicate at encoder rate for RTSP). Full pipeline diagram and queue behavior: **[`PLOTTER_PIPELINE.md`](PLOTTER_PIPELINE.md)**.

**Grid axes (video and API):**

- **Horizontal** = **symbol index** `s` from **0** to **`symbols − 1`** (left → right).
- **Vertical** = **PRB index** `p` from **0** at the **bottom** to **`prb − 1`** at the **top** (frequency-style orientation).

This matches the Vite IQ axes (`iq_symbols`, `iq_prb` from `/api/plotter/status`).

---

## 2. FIFO paths vs in-process frame queues

Integrators sometimes confuse **named pipes on disk** with the plotter’s **in-memory** queues.

**Named FIFOs (real paths; backend `mkfifo` on Start):**

| Pipe | Environment variable | Default path |
|------|---------------------|--------------|
| Full-grid I/Q | **`CUDA_IQ_FIFO_PATH`** | **`/tmp/resource_grid.fifo`** |
| Slot allocation messages | **`CUDA_SLOT_FIFO_PATH`** | **`/tmp/fapi.fifo`** |
| Raw RGBA video (`--fifo-images`) | **`RESGRID_FIFO_IMAGES`** | **`/tmp/images.fifo`** (when unset in env; empty string disables) |

These are passed to **`resgrid_plotter`** as **`--fifo-resource-grid`**, **`--fifo-fapi`**, and **`--fifo-images`** respectively.

**Ownership after `mkfifo`:** When the backend creates a pipe on a **host-mounted** directory (e.g. **`resouce-grid/host-tmp` → `/tmp`** in Compose), it runs **`chown`** so non-root peers can open it. **`RESGRID_FIFO_CHOWN`** defaults to **`1000:1000`** (`uid:gid`). Set **`RESGRID_FIFO_CHOWN_DISABLE=1`** to skip. If pipes were created outside the backend, run **`scripts/chown_resgrid_fifos.sh`** on that directory.

When running **two** backends (DL + UL, **`compose/docker-compose.dlul.yaml`**), use **distinct** FIFO paths per container (defaults such as **`/tmp/resource_grid_dl.fifo`**, **`/tmp/fapi_ul.fifo`**, etc., or the **`CUDA_*_PATH_DL` / `_UL`** variables in that compose file) so external tools can target one direction without colliding on the same pipe names.

### Dual DL / UL (`compose/docker-compose.dlul*.yaml`)

Two **FastAPI** + **`resgrid_plotter`** publishers run in parallel and share one **MediaMTX** and one **Vite** app (two columns in the UI).

| Concern | Downlink (DL) | Uplink (UL) |
|---------|----------------|-------------|
| Default host API port (compose) | **8081** | **8082** |
| Compose service name | `backend-dl` | `backend-ul` |
| MediaMTX path (HLS / WHEP segment) | **`live_dl`** | **`live_ul`** |
| Browser env (`VITE_*`) | **`VITE_API_BASE_DL`** | **`VITE_API_BASE_UL`** |

- **MediaMTX config:** **`mediamtx/mediamtx.dlul.yml`** (not the single-path **`mediamtx.yml`**). Each backend’s **`MTX_RTSP_URL`** targets **`…/live_dl`** or **`…/live_ul`** as in **`compose/docker-compose.dlul.yaml`**.
- **Vite / same-origin proxy:** DL/UL compose sets **`VITE_API_RELATIVE=0`** and does **not** use **`VITE_DEV_PROXY_TARGET`** — the UI issues **`fetch()`** and **OpenAPI /docs** links against the two **absolute** API bases. Clients must be able to reach **both** API ports (and **5173**, **8888**, etc.) from the browser, unless you put a gateway in front and set **`VITE_API_BASE_DL`** / **`VITE_API_BASE_UL`** to those URLs.
- **OpenAPI / Swagger:** Each column’s **OpenAPI /docs** link uses that column’s **`VITE_API_BASE_*`** (DL → **8081**, UL → **8082**). If those env values use **`localhost`** / **`127.0.0.1`** but you open the UI at a LAN IP, the frontend rewrites the host to **`window.location.hostname`** (**`frontend/src/config/panelLayout.ts`**).
- **FIFO paths:** Defaults are **direction-specific** in compose (**`CUDA_IQ_FIFO_PATH_DL` / `_UL`**, **`CUDA_SLOT_FIFO_PATH_DL` / `_UL`** → **`CUDA_IQ_FIFO_PATH`** / **`CUDA_SLOT_FIFO_PATH`** inside each container); keep DL and UL pipes on the same host bind-mount if a host process must choose which pipe to open.

Compose files: **`compose/docker-compose.dlul.yaml`** (default **273** PRB per backend), **`compose/docker-compose.dlul.bw106.yaml`** (**106**), **`compose/docker-compose.dlul.bw273.yaml`** (**273** naming variant).

**OAI gNB writer:** If **`resgrid_plotter`** or the backend **restarts**, the FIFO read end closes and the gNB may see **`EPIPE` / Broken pipe** on `write`. Current OAI MAC code **closes the writer fd** on write failure and **re-opens** on the next slot so connectivity recovers when the reader is back (see **`openairinterface5g/.../resgrid_plotter_slot_fifo.c`**).

**In-process queues — not files:** **`CUDA_IQ_QUEUE_DEPTH`** (default **`32`**) maps to **`--queue-depth`** and caps how many complete **`GridFrame`** / **RGBA** items can wait inside the binary. There is **no** “queue file name” for this knob. When a bounded queue is full, **`push` drops the oldest** item and enqueues the new one (counters such as **`g_grid_frame_queue_drops`**, **`g_rgba_frame_queue_drops`**, **`g_slot_completed_drops`**, … in **`plotter_common.cuh`** / **`plotter_management.cu`**). The optional **file capturer** queue uses the same **drop-oldest** policy. See **[`PLOTTER_PIPELINE.md`](PLOTTER_PIPELINE.md)** §6.

- **`GridFrameQueue`** (internal synth, full-grid IQ reader, channel-mode IQ sidecar): when full, **drops oldest** then **push** ( **`g_grid_frame_queue_drops`** ).
- **`SlotCompletedQueue`** (grids **finished** when **`slot_id`** advances): when full, **drops oldest** ( **`g_slot_completed_drops`** ).
- **`RgbaFrameQueue`**, **`RasterWorkQueue`**, **`CapturerFrameQueue`**: same **drop-oldest** pattern when full (see plotter sources for **`g_*_drops`**).

Source: **`backend/plotter/include/plotter_common.cuh`** (`GridFrameQueue`, `SlotCompletedQueue`, `InternalFrameQueue`, `FullGridFifoQueue`, `ChannelModeIqSidecarQueue`).

---

## 3. Flat index and I/Q layout

For a grid with **`symbols`** OFDM symbols and **`prb`** PRBs, **`ncells = symbols * prb`**.

**Row-major by symbol** (symbol outer, PRB inner):

```text
flat_index(s, p) = s * prb + p     // 0 <= s < symbols, 0 <= p < prb
```

Each cell is **one** complex sample:

- **`float32` real**, **`float32` imag**, **little-endian**, interleaved as **`re, im`** per cell.

**One IQ FIFO “frame”** (when using `--fifo-resource-grid`) is **`ncells * 8`** bytes: cells in order **`flat_index(0,0), flat_index(0,1), …, flat_index(symbols−1, prb−1)`**.

The **slot FIFO payload** (§5) uses the **same** indexing for its rectangle, but with an offset derived from **`sym_start`**, **`prb_start`** (§6 payload indexing).

---

## 4. Logical channel ids and labels

Ids are grouped as **DL↔UL pairs** for easier recall (sync/access, control, data, sounding):

| Id | Overlay name | Typical pairing |
|----|----------------|------------------|
| **0** | — | Viridis only (magnitude heatmap). |
| **1** | SSB | With **PRACH** (initial access). NFAPI **`SSB`** PDU geometry (PSS/SSS/PBCH block). |
| **2** | PRACH | With **SSB**. |
| **3** | PDCCH | With **PUCCH** (control). |
| **4** | PUCCH | With **PDCCH**. |
| **5** | PDSCH | With **PUSCH** (data). |
| **6** | PUSCH | With **PDSCH**. |
| **7** | CSI-RS | With **SRS** (sounding); neutral gray tint until a distinct palette is chosen. |
| **8** | SRS | With **CSI-RS**; neutral gray tint today. |

**Compatibility:** Older slot FIFO captures used **`1`** for SSB (labeled PBCH), **`2=PDCCH`, `3=PDSCH`, `4=PUSCH`, `5=PUCCH`, `6=PRACH`** — replotting those raw **`channel`** bytes with this layout **without regenerating** from OAI will show **wrong** PHY colors and labels.

**Rendering:** CUDA applies **`channel_tint_rgb(id)`** as the base for ids **1…8**; when per-cell **`cell_rnti`** is present and non-zero, a small deterministic RGB jitter is added (**`channel_tint_modulated`** in `plotter_raster.cu`) so the same PHY channel can look slightly different per RNTI (e.g. two PDCCH or two PUSCH allocations). Magnitude is not used for that tint. Id **0** uses **Viridis** on magnitude from the I/Q grid (slot **shape-only** patches leave zeros there → dark). Naming matches **`chan_phy_name()`** (C++) and **`generator.py`** demos.

**Region outlines:** Before centroid labels, the plotter strokes each PHY cell’s edges in screen space where the neighbor is not the same **`chan`** / **`cell_rnti`** pair (or is outside the grid), so each allocation’s bounding box aligns with the same discrete (**symbol** × **PRB**) mapping as **`rgba_frame_kernel`**.

**Region labels:** For each id **1…8** present in `chan`, the plotter draws a **two-line** label near the **centroid** of each **4-connected** component: **PHY** short name, then an **RNTI** line as **`0x` + uppercase hex** when per-cell **`cell_rnti`** is populated (**slot FIFO** scatters each patch’s `rnti`). Neighbors merge only if **`chan` matches *and* `cell_rnti` matches**, so two regions with the same PHY but different RNTIs get **separate** labels (all logical channels **1…8**, not only PDSCH). If **`cell_rnti`** is empty, only the PHY line is drawn (merge by channel only). If **all** `chan` cells are **0**, this step is skipped.

---

## 5. Slot FIFO wire format (Path C)

**Pipe:** Named FIFO (default path `CUDA_SLOT_FIFO_PATH`, e.g. `/tmp/fapi.fifo`). The backend creates it when starting **slot-message / channel-based** streaming (`SYNTHETIC_FAPI` / historic **`SYNTHETIC_CHANNEL_BASED`**, or **`CUDA_PLOTTER_DATA_MODE=3`**), and passes **`--fifo-fapi`** to **`resgrid_plotter`**. The legacy **`CUDA_SLOT_USE_FIFO=1`** flag is **ignored** when the plotter runs in **full-grid FIFO** (`--data-mode 2`); use **SYNTHETIC_FAPI** (slot path) or **`CUDA_PLOTTER_DATA_MODE=3`** to enable slot FIFO.

### 5.1 Version 1 (single rectangle per record)

**One record:**

1. **Fixed 40-byte header** (`#pragma pack(1)`, **little-endian**).
2. **Optional** **`n_complex` × 8** bytes: interleaved **`float32 re, im`**, same as IQ FIFO. If **`n_complex = 0`** but **`sym_count * prb_count > 0`**, the record is **shape-only**: no payload follows; the plotter paints **`channel` / `rnti`** over the rectangle and **flat PHY tint** (ids **1…8**), leaving I/Q samples as zero in that patch. If **`n_complex > 0`**, it must equal **`sym_count * prb_count`** and the full payload must be present (**legacy** behaviour).

**Header layout** (40 bytes total):

| Offset | Type | Field | Notes |
|--------|------|--------|--------|
| 0 | `uint32` | `magic` | Must be **`0x31564153`**. |
| 4 | `uint16` | `version` | **`1`** (single rectangle in header). |
| 6 | `uint16` | `flags` | **`0`** for v1. |
| 8 | `uint32` | `slot_id` | Slot boundary key (§6). |
| 12 | `uint16` | `sym_start` | First symbol `s` (inclusive). |
| 14 | `uint16` | `sym_count` | Number of symbols in patch (may be **0**). |
| 16 | `uint16` | `prb_start` | First PRB `p` (inclusive). |
| 18 | `uint16` | `prb_count` | Number of PRBs in patch (may be **0**). |
| 20 | `uint32` | `rnti` | Stored per cell in `cell_rnti` for centroid labels when `chan` **1…8**. |
| 24 | `uint8` | `channel` | Logical channel id written into `chan` for every cell in the patch. |
| 25 | `uint8` | `reserved0` | **0** |
| 26 | `uint16` | `reserved1` | **0** |
| 28 | `uint32` | `n_complex` | **0** = shape-only (no I/Q bytes); **else** must equal **`sym_count * prb_count`** (payload follows). |
| 32 | `uint32` | `reserved2` | **0** |
| 36 | `uint32` | `reserved3` | **0** |

### 5.2 Version 2 — multiple rectangles in **one** record (`flags = 1`)

Use this when a **single slot** should carry **several** independent allocations, each with its own **`sym_start` / `sym_count` / `prb_start` / `prb_count`**, **`channel`**, and **`rnti`**, in **one** FIFO write.

**Record layout:**

1. **40-byte header** with **`version = 2`**, **`flags = 1`** (**`kSlotFlagMultiRegion`**).
2. **`uint16_t` `n_regions`** — number of descriptors (**1…512**).
3. **`n_regions` × 16-byte** **`SlotRegionDesc`** (packed, little-endian), in processing order.
4. **Payload:** **`n_complex` × 8** bytes — concatenation of each region’s I/Q, in the **same order** as the descriptors. Per region, layout matches §6 (row-major: symbol outer, PRB inner). Omit the payload entirely when **`n_complex = 0`** (shape-only multi-region: **Σ** region cells may be **> 0**).

**Header fields for v2 multi:** Set **`sym_start` / `sym_count` / `prb_start` / `prb_count` / `rnti` / `channel`** to **0** (ignored). **`n_complex`** must equal **Σ** (`sym_count` × `prb_count`) over all descriptors when carrying I/Q; use **`n_complex = 0`** for shape-only records (same descriptor layout, no float payload).

**`SlotRegionDesc`** (16 bytes):

| Offset | Type | Field |
|--------|------|--------|
| 0 | `uint16` | `sym_start` |
| 2 | `uint16` | `sym_count` |
| 4 | `uint16` | `prb_start` |
| 6 | `uint16` | `prb_count` |
| 8 | `uint32` | `rnti` |
| 12 | `uint8` | `channel` |
| 13 | `uint8` | `reserved[3]` | **0** |

**Python:** `struct.pack("<HHHHIB3x", …)` per region; extension = `pack("<H", n_regions) + b"".join(regions)`. See **`backend/scripts/generator.py`** (`pack_slot_region`, `--slot-pattern demo-multi`).

**C++:** `struct SlotRegionDesc` in **`backend/plotter/include/plotter_common.cuh`**.

### 5.3 Version 2 — single rectangle (optional)

**`version = 2`** and **`flags = 0`**: same semantics as **v1** — one rectangle from the 40-byte header (no extension). Useful if you standardize on v2 but do not need multi-region.

**Python reference (header):** `struct.pack("<IHHIHHHHIBBHIII", ...)` as in **`backend/scripts/generator.py`** (`SLOT_HDR_FMT` / `pack_slot_hdr`).

**C++ reference:** `struct SlotMsgHdr` in `backend/plotter/include/plotter_common.cuh`.

---

## 6. Slot assembly semantics

The plotter maintains one **active** `slot_id` and a **full-size** complex grid + **`chan`** plane (zeros initially).

- **First message** in a session sets **`active_slot = slot_id`** and **scatters** payload(s) and **`channel` / `rnti`** into each rectangle (v1: one rectangle from header; **v2 multi:** one pass per **`SlotRegionDesc`** in order).
- **Same `slot_id`:** Further messages **update** the same grid; **overlapping cells: last write wins** (including within one v2 multi record: later regions overwrite earlier on overlap).
- **Greater `slot_id`:** The **previous** grid is **finalized** (queued for the normal CUDA → NVENC path), buffers are cleared, then the new message applies to the **new** slot.
- **Smaller `slot_id`** than `active_slot` (**non-monotonic**): message is **dropped** (rate-limited **stderr** warning).

**Payload indexing** for each rectangle with valid `sym_count`, `prb_count` > 0:

```text
global_idx = (sym_start + s) * prb + (prb_start + p)   // s in [0, sym_count), p in [0, prb_count)
payload_index = s * prb_count + p                       // row-major: s outer, p inner
```

**Shutdown / EOF:** The reader **flushes** the open slot (pushes one final grid) so the last assembly is not lost.

**Encode priority:** A queued **completed slot** frame is consumed **before** the next `--fifo-resource-grid` read or built-in synthetic frame for that iteration. Sidecar IQ messages carry **I/Q only** (`chan` cleared in the reader); the plotter **reuses the last completed slot’s** `chan` / `cell_rnti` on those frames so the video does not **flash** between PHY tints and bare Viridis when IQ arrives faster than slot boundaries.

**Target pipeline layout:** **Assembler**, **Raster**, and **Encoder** may be split into explicit stages (completed-slot queue → image queue → NVENC); see **[`PLOTTER_PIPELINE.md`](PLOTTER_PIPELINE.md)** (*Target architecture: Assembler, Raster, Encoder*).

---

## 7. Ways to supply channel coloring

| Approach | Per-cell `chan` | Typical use |
|----------|------------------|-------------|
| **`--fifo-fapi`** (slot-message / **`--data-mode 3`**) | Filled from each message’s **`channel`** (and **`rnti`**) over each patch; **v2 multi** applies several patches per FIFO record | **Only** source of PHY tints / RNTI in normal operation. |
| Built-in C++ producer (`CUDA_IQ_USE_FIFO=0`, no slot) | **All 0** | Viridis-only heatmap; enable **slot FIFO** (or Python demo with `--fifo-fapi`) for channel coloring. |
| **`--fifo-resource-grid` only** | **All 0** (Python producer sends I/Q only) | Magnitude-only Viridis unless **slot FIFO** also supplies `chan`. |

---

## 8. Control shared memory (`--cuda-shm`)

The backend creates an **`88`‑byte** POSIX shared memory object passed as **`--cuda-shm <name>`**. **`resgrid_plotter`** updates rates and reads commands/pattern.

| Offset | Size | Type | Content |
|--------|------|------|---------|
| 0 | 8 | `double` | **Built-in producer** path (`--fifo-resource-grid` off): measured synthetic producer rate (Hz), updated ~1×/s. With **`--fifo-resource-grid`**, this slot is **not** refreshed by the stats thread (often stays the initial NaN). |
| 8 | 8 | `double` | Measured **Raster** RGBA output rate (Hz), updated ~1×/s. |
| 16 | 4 | `uint32` | Synthetic **pattern** id (**0…3**); backend/API can set; plotter reads for **built-in I/Q only** (waves / tone / Gaussian / vortex). Does **not** paint PHY channels — use slot FIFO for that. |
| 20 | 4 | — | *Padding* (uninitialized zeros in the backend initializer; not read by the plotter for logic). |
| 24 | 8 | `double` | **Commanded** frame rate / producer Hz; backend writes; plotter reads for pacing. |
| 32 | 8 | `uint64` | Cumulative **published frames** (incremented after each full write to FFmpeg stdin). |
| 40 | 48 | six × `double` | **Assembler** in + out (Hz) measured; **Raster in** = **Assembler out**, **Raster out** = offset **8**; **Encoder in** = **Raster out**; **Encoder out** = NVENC stdin writes/s (chained telemetry). See [`PLOTTER_PIPELINE.md`](PLOTTER_PIPELINE.md) §9. |

*Exact packing matches* `backend/app/main.py` (`_pack_cuda_shm_init`, `_IQ_CUDA_SHM_BYTES = 88`) and `FpsShmMap` in `plotter_management.cu`.

When **`CUDA_IQ_USE_FIFO=1`**, a separate **`32`‑byte** **`--generator-shm`** block (alias **`--producer-shm`**) is created for the Python IQ generator (measured Hz, commanded Hz, pattern) — see the docstring in `generator.py`.

---

## 9. Environment variables (summary)

| Variable | Role |
|----------|------|
| `GENERATION_MODE` | **`SYNTHETIC_FAPI`** (or env `synthetic_channel_based`) names the channel-oriented demo mode (same runtime as full-grid **`SYNTHETIC_RESOURCE_GRID`** for the CUDA binary); PHY colors still require **slot FIFO**. **`CHANNEL_BASED_SYNTHETIC_GENERATION`** remains accepted. |
| `CUDA_IQ_USE_FIFO` | `1` → IQ FIFO + optional `generator.py`. |
| `CUDA_IQ_FIFO_PATH` | IQ named pipe path (default **`/tmp/resource_grid.fifo`**; see §2). |
| `CUDA_IQ_QUEUE_DEPTH` | In-process **`GridFrame`** deque capacity (**not** a file); passed as **`--queue-depth`** (default **`32`**). See §2. |
| `CUDA_SLOT_USE_FIFO` | **Deprecated / ignored** with full-grid FIFO (`--data-mode 2`). Use **`SYNTHETIC_CHANNEL_BASED`** or **`CUDA_PLOTTER_DATA_MODE=3`** for slot FIFO. (Still present in some compose files for backward compatibility.) |
| `CUDA_SLOT_FIFO_PATH` | Slot named pipe path (default **`/tmp/fapi.fifo`**; see §2). |
| `RESGRID_FIFO_CHOWN` | **`uid:gid`** applied after **`mkfifo`** on IQ/slot pipes (default **`1000:1000`**). |
| `RESGRID_FIFO_CHOWN_DISABLE` | **`1`** / **`true`** → do not **`chown`** FIFOs. |
| `CUDA_IQ_PRB`, `CUDA_IQ_SYMBOLS` | Must match your writers and plotter args. Compose **variants** set different **`CUDA_IQ_PRB`** defaults (**273** / **133** / **106**); see [`ARCHITECTURE.md`](ARCHITECTURE.md) **Configuration reference**. |
| `CUDA_SLOT_WINDOW_MIN_PRB` | Default **50** (backend + generator). Minimum PRB span per OFDM symbol for **`symbol-windows`** slot demo (**`--slot-window-min-prb`**); capped by grid **`--prb`**. |

Full list: **`.env.example`** and [`ARCHITECTURE.md`](ARCHITECTURE.md) (IQ / FIFO env table). **FIFO paths vs `--queue-depth`:** §2 above.

---

## 10. Reference tools

- **`backend/scripts/generator.py`** — Full-grid IQ FIFO producer at **`CUDA_IQ_GEN_FPS`**; optional **`--generator-shm`** / **`--producer-shm`** for pattern and Hz from the backend. With **`--fifo-fapi`**, emits demo slot records on **`--slot-interval`**; use **`--slot-only`** for slot-FIFO-only runs. **CLI defaults when flags are omitted:** **`--prb`** ← env **`CUDA_IQ_PRB`** (else **273**); **`--slot-window-min-prb`** ← **`CUDA_SLOT_WINDOW_MIN_PRB`** (else **50**) — same env vars the FastAPI backend uses when it spawns this script.
- **REST** — `GET /api/plotter/status` returns `iq_prb`, `iq_symbols`, mode, and CUDA **stats** when running.

---

## 11. Versioning

- **Slot header `version`:** **`1`** = single rectangle in header. **`2`** = same when **`flags = 0`**, or **multi-region** when **`flags = 1`** (extension §5.2).
- **`flags`:** bit **0** = **`kSlotFlagMultiRegion`** (**`1`**) → **`n_regions`** + **`SlotRegionDesc[]`** after header.
- **`magic`:** **`0x31564153`**; wrong magic → record skipped.

If you extend the protocol further, keep **little-endian** packing and update this doc together with `SlotMsgHdr` / `SlotRegionDesc` / **`generator.py`** (slot pack helpers).
