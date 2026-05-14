# cuPHY PUSCH uplink → `resgrid_plotter` (POSIX mq)

This stack path exports **PUSCH `DataRx`** (post–front-end resource grid on the GPU) to the same **full-grid I/Q** contract described in [`CHANNEL_BASED_INTERFACE.md`](CHANNEL_BASED_INTERFACE.md) §3 (`float32` re/im, row-major `flat_index(s,p)=s*prb+p`).

Transport is **POSIX message queues** only (Linux), with the **chunked RGMC header** (v2 includes PRB/symbol counts) used by **`resgrid_plotter`** and **`backend/app/resgrid_posix_mq.py`**. Named-pipe / FIFO export was removed from cuPHY.

## Enable in cuPHY / cuBB

| Variable | Meaning |
|----------|---------|
| **`CUPHY_RESGRID_MQ_ENABLE`** | Set to **`1`** / **`true`** / any non-empty non-`0` string to enable. **Default: off** (no worker thread, no extra D2H). |
| **`CUPHY_RESGRID_FIFO_ENABLE`** | **Legacy alias** for **`CUPHY_RESGRID_MQ_ENABLE`** (same truthy rules). Kept for older compose/scripts. |
| **`CUPHY_RESGRID_MQ_PATH`** | POSIX mq name (must start with **`/`**), e.g. **`/resgrid_iq`**. Optional override if **`CUDA_IQ_MQ_PATH`** is unset. |
| **`CUDA_IQ_MQ_PATH`** | Same as plotter / FastAPI: must be set to a **`/`** name for export to run (e.g. **`/resgrid_iq`**). |
| **`CUPHY_RESGRID_PRB`** | Output PRB count for the plotter frame (default: `NF/12` from the tensor). Must match backend **`CUDA_IQ_PRB`**. |
| **`CUPHY_RESGRID_SYMBOLS`** | Output symbol count (default: tensor NT, usually **14**). Must match **`CUDA_IQ_SYMBOLS`**. |
| **`CUPHY_RESGRID_ANTENNA`** | **`0…N_RX-1`**: coherent average over the 12 subcarriers of that antenna. **`-1`**: RMS magnitude across all antennas (re on wire, im **0**). |
| **`CUPHY_RESGRID_ANT0_ONLY`** | **`1`** (default): if the tensor has **N_RX > 1** and **tight** strides `(1, dim0, dim0×dim1)`, D2H copies **only RX antenna 0** (one contiguous plane); the worker sees **dim2 = 1**. **`0`**: copy the full multi-antenna tensor. |

**Cell selection:** export always uses aggregated PUSCH **cell index `0`** only (one logical frame per slot). **`CUPHY_RESGRID_PUSCH_CELL`** is not read.

**Non-Linux:** export is disabled (no-op). **Linux** without **`CUDA_IQ_MQ_PATH`** or **`CUPHY_RESGRID_MQ_PATH`**: the enable flag is ignored for enqueue (stderr hints).

**Datalake / `cpuCopyOn`:** Export uses a **separate async D2H** from **`PuschRx::copyUlToResgridMq`** (prefers **`pDataIn->pTDataRx`**, i.e. the same **UL stage-1** buffer the driver wires to PUSCH) into internal pinned staging. It does **not** require **`pDataRxHost`** / datalake. If **`m_outputPrms.cpuCopyOn`** is false, the driver still runs **`phase2Stream`** waits and invokes **`copyUlToResgridMq`** during **`PUSCH_RUN_FULL_SLOT_COPY`** / **`PUSCH_RUN_ALL_PHASES`** so export works without full CPU decode copies. **`cuphyWriteDbgBufSynch`** may also call **`copyUlToResgridMq`** when export is enabled.

**Real-time behavior:** If the background worker is still converting or writing the previous frame, a new slot **drops** the export (counter increments internally) instead of blocking the PUSCH **`copyOutputToCPU`** path on a mutex wait.

## Docker / Compose alignment

1. Use the **same mq name** as the backend (**`CUDA_IQ_MQ_PATH`**, default **`/resgrid_iq`**). The **backend must create the queue** (`mq_unlink` + `mq_open(O_CREAT|O_RDONLY)` then close) before **`resgrid_plotter`** opens read and before cuPHY opens write—same ordering as **`generator.py`**.
2. Containers must share the **host POSIX mqueue namespace** (or equivalent); bind-mounting **`/tmp`** alone does **not** share mq names.
3. Set **`SYNTHETIC_RESOURCE_GRID`**, **`CUDA_AUTO_START_IQ_GENERATOR=0`**, and pass **`--mq-resource-grid`** to the plotter. Align **`RESGRID_MQ_MSGSIZE`** / **`RESGRID_MQ_MAXMSG`** with **`/proc/sys/fs/mqueue/*`** if frames are large.

Match **PRB** and **symbol** counts between **`CUPHY_RESGRID_*`** and **`CUDA_IQ_PRB` / `CUDA_IQ_SYMBOLS`** so the plotter does not misinterpret frame size.

## Layout sanity script

`backend/scripts/verify_resgrid_fifo_layout.py` checks **`flat_index(s,p)=s*prb+p`** and prints the byte size **`symbols*prb*2*sizeof(float)`** for given **`--prb`** / **`--symbols`** (defaults follow **`CUDA_IQ_PRB`** / **`CUDA_IQ_SYMBOLS`**).

## Soak / timing

Compare PUSCH slot timing or underflow counters with the export **disabled** vs **enabled**. If the worker cannot keep up, expect **dropped** frames, not longer CUDA work on **`phase2Stream`**.

## Troubleshooting (no frames / `mq_open` failures)

1. **Plotter running:** The backend must create the queues and spawn **`resgrid_plotter`** ( **`AUTO_START=1`** in compose, or **`POST /api/plotter/start`** ). Until then, **`mq_open(..., O_WRONLY)`** in cuPHY often fails with **`ENOENT`** (queue does not exist).
2. **Same IPC namespace:** With **`ipc: host`** on the **visualizer `backend`** service, queues live in the **host** namespace. A **cuBB / PHY container** must also use **`ipc: host`** (or share the backend’s IPC namespace), not only the backend.
3. **Exact mq name:** **`CUDA_IQ_MQ_PATH`** (or **`CUPHY_RESGRID_MQ_PATH`**) must match the backend (**`/resgrid_iq`** by default; DL/UL stacks use **`_dl` / `_ul`** suffix names).
4. **Enable export:** **`CUPHY_RESGRID_MQ_ENABLE=1`** (or truthy; legacy **`CUPHY_RESGRID_FIFO_ENABLE`**) must be set in the PHY process.
5. **Host checks:** After starting the backend plotter, on the Linux host run **`ls -la /dev/mqueue/`** — you should see the configured names (e.g. **`resgrid_iq`**). If nothing appears, the backend did not create the queues yet or a different IPC namespace is in use.

**Note:** This file name still contains `FIFO` for historic links; cuPHY now uses **POSIX mq only**.
