# Full-grid I/Q POSIX message queue (`/resgrid_iq` default)

This note describes the **logical payload** (one full-grid I/Q raster) consumed by **`resgrid_plotter`** via **`--mq-resource-grid`** and how it is carried over **POSIX message queues**. The canonical integrator guide is **[`docs/CHANNEL_BASED_INTERFACE.md`](../../docs/CHANNEL_BASED_INTERFACE.md)** (especially §2–§3); this file is a compact plotter-side summary.

---

## 1. What it is (not TCP)

- Transport is a **POSIX message queue**: a kernel object addressed by a **name** starting with **`/`** (default **`/resgrid_iq`**, env **`CUDA_IQ_MQ_PATH`** on the FastAPI stack).
- **Logical framing**: each **logical frame** (one complete grid blob, same byte layout as §3) may be split into **multiple** `mq_send` messages. Every mq message is **chunk header + chunk payload**; the header magic is **`0x52474D43`** (`RGMC`). **Version 1** uses a **24-byte** header; **version 2** extends it to **28 bytes** with **`prb`** and **`symbols`** (`uint16` each) describing the full-grid shape when applicable (RGB / slot producers may leave them **0**). The reader **`ResgridMqStreamIn`** accepts v1 and v2, reassembles chunks into a **byte stream**, then performs **`read_exact`** in **`frame_bytes`** multiples—so producer semantics match the old “one record per frame” model at the **payload** level. See **`readme.md`** § “Chunked wire format” and **`include/plotter_posix_mq.cuh`** / **`src/plotter_posix_mq.cpp`**.
- **Producer-side helpers**: Python **`backend/app/resgrid_posix_mq.py`** (`ResgridMqWriter`) and **`generator.py`** use the same chunking as the C++ **`resgrid_mq_send_frame`** path.
- **Queue limits**: align **`RESGRID_MQ_MSGSIZE`** / **`RESGRID_MQ_MAXMSG`** with **`/proc/sys/fs/mqueue/*`** so chunks are not truncated; defaults follow **`effective_msgsize_cap()`** in **`plotter_posix_mq.cpp`**.

**Standalone “slot-IQ” wire** on the **same** mq (without FAPI) uses a different logical record shape (**`ResourceGridSlotIqHdr`** + patch); see **`readme.md`** and **`plotter_common.cuh`**. This file focuses on **full-grid** mode (**`--mq-resource-grid-mode full`**).

---

## 2. Grid parameters

| Parameter | Meaning |
|-----------|---------|
| **`prb`** | Number of physical resource blocks in the I/Q raster (**width** in PRB dimension). Must match **`CUDA_IQ_PRB`** / plotter **`--prb`**. |
| **`symbols`** | Number of OFDM symbols in the raster (**horizontal** axis in the UI). Must match **`CUDA_IQ_SYMBOLS`** (often **14**). |

**Video / API orientation** (same as the main interface doc):

- **Horizontal** = symbol index **`s`**, **`0 … symbols − 1`** (left → right).
- **Vertical** = PRB index **`p`**, **`0 … prb − 1`** (bottom → top in frequency-style layout).

---

## 3. Binary data format (one logical frame payload)

After mq **reassembly**, the plotter sees a contiguous blob identical to the historic single-writer pipe record.

Let **`ncells = symbols × prb`**.

**Indexing** — row-major with **symbol outer**, **PRB inner**:

```text
flat_index(s, p) = s * prb + p     // 0 <= s < symbols, 0 <= p < prb
```

**Each cell** is one complex sample:

| Field | Type | Notes |
|-------|------|--------|
| **re** | **`float32`**, little-endian | In-phase |
| **im** | **`float32`**, little-endian | Quadrature |

Cells are packed in **increasing `flat_index`** order:  
`(s=0,p=0), (s=0,p=1), …, (s=0,prb−1), (s=1,0), …, (s=symbols−1, prb−1)`.

**One logical frame** has size:

```text
frame_bytes = ncells × 8 = symbols × prb × 2 × sizeof(float)
```

There is **no** metadata plane in this stream (no `chan` / RNTI here). Channel coloring, if needed, comes from the **slot / FAPI** mq path (**`/resgrid_slot`** default, **`--mq-fapi`**) in channel-based modes; see **[`CHANNEL_BASED_INTERFACE.md`](../../docs/CHANNEL_BASED_INTERFACE.md)**.

---

## 4. Reader / writer behaviour (protocol-level)

- The **backend** creates the queue (**`mq_unlink` + `mq_open(O_CREAT|O_RDONLY)`** with attrs, then closes) before spawning **`resgrid_plotter`**, which opens the name **`O_RDONLY`**. Producers open **`O_WRONLY`** and send chunked frames; **`mq_send`** may block when the queue is full (**backpressure**).
- If the **reader** exits, the writer may see **`EPIPE`** (pipes) / **`mq_send`** errors; producers should **close** the descriptor and **re-open** when a reader returns. cuPHY uses **POSIX mq** via **`mq_sink_rx`** (chunked wire); see **`docs/CUPHY_UL_RESGRID_FIFO.md`** for env variables.

---

## 5. Configuration checklist

1. **Same mq name** in plotter argv, backend env, and producer (containers must share the **host mqueue namespace** / compatible deployment; bind-mounting **`/tmp`** does **not** by itself share mq names across hosts).
2. **`prb`** and **`symbols`** must **match** between producer and **`resgrid_plotter`** (**`CUDA_IQ_PRB`**, **`CUDA_IQ_SYMBOLS`** in compose / env).
3. Sanity check frame size: **`backend/scripts/verify_resgrid_fifo_layout.py`** (name is historic) —  
   **`../../docs/CUPHY_UL_RESGRID_FIFO.md`** for **cuPHY** export variables (filename is historic; transport is mq).

---

## 6. See also

| Document | Role |
|----------|------|
| [`docs/CHANNEL_BASED_INTERFACE.md`](../../docs/CHANNEL_BASED_INTERFACE.md) | Slot vs full-grid, flat index, env reference |
| [`docs/CUPHY_UL_RESGRID_FIFO.md`](../../docs/CUPHY_UL_RESGRID_FIFO.md) | cuPHY PUSCH `DataRx` export variables (POSIX mq; legacy doc filename) |
| [`readme.md`](readme.md) | Plotter pipeline, **`--mq-*`** argv groups, chunked header |

**Note:** This file is named `readme.resouce_grid.md` (historic spelling). The default **POSIX mq** name is **`/resgrid_iq`**, controlled by **`CUDA_IQ_MQ_PATH`**.
