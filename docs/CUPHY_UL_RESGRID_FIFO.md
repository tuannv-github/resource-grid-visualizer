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

### Optional NAI export (second POSIX mq)

Independent of the UL export above: a **second** worker thread writes **`cuphy_resgrid_mq_enqueue_nai_from_device`** to **`CUDA_IQ_MQ_PATH_NAI`** or **`CUPHY_NAI_RESGRID_MQ_PATH`**. Use a **different** mq name than UL (e.g. **`/resgrid_iq_nai`**) and a second FastAPI/plotter stack (see **`compose/docker-compose.ulnai.bw133.yaml`**).

| Variable | Meaning |
|----------|---------|
| **`CUPHY_NAI_RESGRID_MQ_ENABLE`** | **`1`** / truthy enables NAI export (separate from **`CUPHY_RESGRID_MQ_ENABLE`**). |
| **`CUDA_IQ_MQ_PATH_NAI`** | POSIX mq name for NAI (must start with **`/`**), e.g. **`/resgrid_iq_nai`**. |
| **`CUPHY_NAI_RESGRID_MQ_PATH`** | Optional override if **`CUDA_IQ_MQ_PATH_NAI`** is unset. |
| **`CUPHY_NAI_RESGRID_PRB`** / **`CUPHY_NAI_RESGRID_SYMBOLS`** / **`CUPHY_NAI_RESGRID_ANTENNA`** | Optional; if unset, the NAI worker falls back to the same **`CUPHY_RESGRID_*`** / **`CUDA_IQ_*`** values as UL. |
| *(PUSCH NAI)* | **`tx̂`** on data REs is always rebuilt from **`pTbPayloadsDevice`** (decoded bits) before **`H·tx̂`** when payloads are available. |

**NAI PHY behavior (full-cell isolation, cell 0):** **`cuphyUlNaiSlotBegin`** copies full-cell **`DataRx`**. Each enabled UL channel subtracts on its REs only; elsewhere **`nai = rx`**. **PUSCH** uses **`pusch_nai_grid`** (`H`, **`DataEq`**). **PUCCH/PRACH/SRS** use allocation masks (see env below). **`cuphyUlNaiFinalizeExport`** runs after PUSCH+PUCCH+PRACH+SRS complete. UL mq remains raw **`DataRx`**.

| Variable | Meaning |
|----------|---------|
| **`CUPHY_NAI_CELL_IDX`** | Target cell index (default **0** = radio 0). |
| *(none)* | All PUSCH UE groups on `CUPHY_NAI_CELL_IDX` are subtracted (no per-group env). |
| **`CUPHY_NAI_RESGRID_ANTENNA`** | RX index for export (**`0…N_RX-1`**, default **0**); D2H copies that antenna plane only. **`-1`**: RMS over all RX (full tensor D2H). |
| **`CUPHY_NAI_PUSCH_ENABLE`** / **`PUCCH`** / **`PRACH`** / **`SRS`** | Per-channel subtract (**1** default). Set **0** to skip. |

Tensor layouts: [`oai-nvidia/cuBB/cuPHY/docs/PUSCH_NAI_TENSOR_LAYOUTS.md`](../../oai-nvidia/cuBB/cuPHY/docs/PUSCH_NAI_TENSOR_LAYOUTS.md).

**Compose (UL+NAI):** Match **`CUDA_IQ_MQ_PATH=/resgrid_iq_ul`** and **`CUDA_IQ_MQ_PATH_NAI=/resgrid_iq_nai`** on cuBB and the visualizer (`compose/docker-compose.ulnai.bw133.yaml`). Example cuBB: [`gh81_compal13/docker-compose.yaml`](../../gh81_compal13/docker-compose.yaml). Start plotters with **`AUTO_START=1`** or **`POST /api/plotter/start`** on **:8082** (UL) and **:8083** (NAI).

**Cell selection:**

**Non-Linux:** export is disabled (no-op). **Linux** without **`CUDA_IQ_MQ_PATH`** or **`CUPHY_RESGRID_MQ_PATH`**: the enable flag is ignored for enqueue (stderr hints).

**Datalake / `cpuCopyOn`:** Export uses a **separate async D2H** from **`PuschRx::copyUlToResgridMq`** (prefers **`pDataIn->pTDataRx`**, i.e. the same **UL stage-1** buffer the driver wires to PUSCH) into internal pinned staging. It does **not** require **`pDataRxHost`** / datalake. If **`m_outputPrms.cpuCopyOn`** is false, the driver still runs **`phase2Stream`** waits and invokes **`copyUlToResgridMq`** / **`copyNaiToResgridMq`** during **`PUSCH_RUN_FULL_SLOT_COPY`** / **`PUSCH_RUN_ALL_PHASES`** so export works without full CPU decode copies. **`cuphyWriteDbgBufSynch`** may also call these when the respective export flags are enabled.

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
6. **`OSError: [Errno 24]`** on **`POST /api/plotter/start`**: Usually **`RLIMIT_MSGQUEUE`** (Docker default **819200** bytes ≈ **9** queues at **10×8192**), not **`RLIMIT_NOFILE`**. UL+NAI compose sets **`ulimits.msgqueue: 16777216`** on **`backend-ul`** / **`backend-nai`** — recreate those containers after changing compose. With **cuBB/plotters stopped**, prune stale names: **`ls /dev/mqueue`**; **`mq_unlink`** unused **`resgrid_*`** (legacy **`/resgrid_iq`** without **`_ul`** / **`_nai`** suffixes are common leftovers). Raise **`sysctl fs.mqueue.queues_max`** only if **`queues_max`** is the bottleneck.

**Note:** This file name still contains `FIFO` for historic links; cuPHY now uses **POSIX mq only**.
