#pragma once
/** Shared types, queues, wire format, SHM helpers, and cross-TU declarations for `resgrid_plotter`. */
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#if defined(_WIN32)
#include <string.h>
#else
#include <strings.h>
#endif

/** How pipeline RGBA frames are reduced to ``--encoder-fps`` for NVENC (see ``--encode-policy``). */
enum class EncodePolicy { Last, Mean, FfmpegFilter };

inline int str_ieq(const char* a, const char* b) {
#if defined(_WIN32)
  return _stricmp(a, b) == 0;
#else
  return strcasecmp(a, b) == 0;
#endif
}

inline EncodePolicy parse_encode_policy(const char* av0, const char* s) {
  if (!s || !*s || str_ieq(s, "last")) return EncodePolicy::Last;
  if (str_ieq(s, "mean")) return EncodePolicy::Mean;
  if (str_ieq(s, "ffmpeg") || str_ieq(s, "ffmpeg-filter")) return EncodePolicy::FfmpegFilter;
  fprintf(stderr, "%s: unknown --encode-policy %s (use last|mean|ffmpeg); using last\n", av0, s);
  return EncodePolicy::Last;
}

extern std::atomic<bool> g_run;
extern std::atomic<uint64_t> g_producer_frames;
/**
 * Per-stage output statistics: increment **once** each time that stage **finishes** one output unit
 * (Assembly → one `GridFrame` to downstream; Raster → one RGBA frame; Encoder → one full NVENC stdin write).
 * The pipeline stats thread samples these at ~1 s intervals and publishes **Δcount / Δt** as Hz (see `plotter_management.cu`).
 */
extern std::atomic<uint64_t> g_stage_output_count_assembly;
extern std::atomic<uint64_t> g_stage_output_count_raster;
extern std::atomic<uint64_t> g_stage_output_count_encoder;
extern std::atomic<uint64_t> g_slot_completed_drops;
extern std::atomic<uint64_t> g_grid_frame_queue_drops;
extern std::atomic<uint64_t> g_rgba_frame_queue_drops;
extern std::atomic<uint64_t> g_raster_work_queue_drops;
extern std::atomic<uint64_t> g_capturer_frame_queue_drops;
extern std::atomic<uint64_t> g_assembly_input_msgs;

/** Ids for rate-limited drop warnings (`plotter_warn_queue_drop_oldest`). */
enum class PlotterQueueDropKind : uint32_t {
  GridFrame = 0,
  SlotCompleted,
  RgbaFrame,
  RasterWork,
  CapturerFrame,
  Count_
};

/**
 * Stderr when a bounded queue drops the oldest item(s) while accepting new input.
 * Throttled (mutex + ~2.5 s minimum per `kind`) so concurrent producers do not flood the log; `drops_this_push` counts this push.
 */
void plotter_warn_queue_drop_oldest(PlotterQueueDropKind kind, const char* queue_name,
                                    const char* producer_stage, uint32_t drops_this_push);

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s\n", what, cudaGetErrorString(e));
    std::abort();
  }
}

/** sRGB EOTF / OETF (IEC 61966-2-1); shared by encode mean policy. */
inline float plotter_srgb_u8_to_linear_f(uint8_t v) {
  const float s = (float)v * (1.0f / 255.0f);
  if (s <= 0.04045f) return s * (1.0f / 12.92f);
  return std::pow((s + 0.055f) * (1.0f / 1.055f), 2.4f);
}

inline uint8_t plotter_linear_f_to_srgb_u8(float x) {
  if (x <= 0.f) return 0;
  if (x >= 1.f) return 255;
  const float s = (x <= 0.0031308f) ? (12.92f * x) : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
  if (s <= 0.f) return 0;
  if (s >= 1.f) return 255;
  return (uint8_t)std::lrint(s * 255.0f);
}

struct GridFrame {
  std::vector<float2> iq;
  std::vector<uint8_t> chan;
  /** Per-cell RNTI when present (slot FIFO); empty => region labels are PHY name only. */
  std::vector<uint32_t> cell_rnti;
};

/** Last completed slot's per-cell PHY metadata; reapplied to IQ sidecar frames so display does not alternate tint vs Viridis. */
struct SlotChanRntiTemplate {
  std::vector<uint8_t> chan;
  std::vector<uint32_t> cell_rnti;
};

class GridFrameQueue {
 public:
  explicit GridFrameQueue(int capacity) : cap_(capacity) {}

  void push(GridFrame&& frame, const char* producer_stage = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!g_run.load(std::memory_order_relaxed)) return;
    uint32_t ndrop = 0;
    while (cap_ >= 1 && (int)q_.size() >= cap_) {
      q_.pop_front();
      ++ndrop;
      g_grid_frame_queue_drops.fetch_add(1, std::memory_order_relaxed);
    }
    if (ndrop)
      plotter_warn_queue_drop_oldest(PlotterQueueDropKind::GridFrame, "GridFrameQueue", producer_stage, ndrop);
    q_.push_back(std::move(frame));
    not_empty_.notify_one();
  }

  bool pop(GridFrame& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!not_empty_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                             [&] { return !q_.empty() || !g_run.load(); }))
      return false;
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

  bool try_pop(GridFrame& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::deque<GridFrame> q_;
};

/** Completed full-grid frames from slot FIFO. */
class SlotCompletedQueue {
 public:
  explicit SlotCompletedQueue(int capacity) : cap_(std::max(1, capacity)) {}

  void push(GridFrame&& frame, const char* producer_stage = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    uint32_t ndrop = 0;
    while ((int)q_.size() >= cap_) {
      q_.pop_front();
      ++ndrop;
      g_slot_completed_drops.fetch_add(1, std::memory_order_relaxed);
    }
    if (ndrop)
      plotter_warn_queue_drop_oldest(PlotterQueueDropKind::SlotCompleted, "SlotCompletedQueue", producer_stage, ndrop);
    q_.push_back(std::move(frame));
    not_empty_.notify_one();
  }

  bool try_pop(GridFrame& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  /** Block up to wait_ms for a grid completed from slot messages (channel-based pipeline, no IQ sidecar). */
  bool pop_wait(GridFrame& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!not_empty_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                             [&] { return !q_.empty() || !g_run.load(); }))
      return false;
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_empty_;
  std::deque<GridFrame> q_;
};

/** Pipeline 1: internal synthetic producer → this queue → interpolate (CUDA) → NVENC. */
using InternalFrameQueue = GridFrameQueue;
/** Pipeline 2 (full-grid FIFO): one `--iq-fifo` read = one complete resource-grid I/Q raster. */
using FullGridFifoQueue = GridFrameQueue;
/** Pipeline 3: slot-message assembler pushes one fully assembled grid (from slot allocations) here. */
using AssembledSlotQueue = SlotCompletedQueue;
/** Pipeline 3 optional: `--iq-fifo` full-grid sidecar while slot messages supply channel metadata / patches. */
using ChannelModeIqSidecarQueue = GridFrameQueue;

/** Raster → Encode: completed RGBA frames (host memory), same backpressure pattern as `GridFrameQueue`. */
class RgbaFrameQueue {
 public:
  explicit RgbaFrameQueue(int capacity) : cap_(std::max(1, capacity)) {}

  void push(std::vector<uint8_t>&& frame, const char* producer_stage = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!g_run.load(std::memory_order_relaxed)) return;
    uint32_t ndrop = 0;
    while ((int)q_.size() >= cap_) {
      q_.pop_front();
      ++ndrop;
      g_rgba_frame_queue_drops.fetch_add(1, std::memory_order_relaxed);
    }
    if (ndrop)
      plotter_warn_queue_drop_oldest(PlotterQueueDropKind::RgbaFrame, "RgbaFrameQueue", producer_stage, ndrop);
    try {
      q_.push_back(std::move(frame));
    } catch (const std::bad_alloc&) {
      fprintf(stderr, "resgrid_plotter: RgbaFrameQueue OOM; stopping pipeline\n");
      g_run.store(false, std::memory_order_relaxed);
      not_empty_.notify_all();
      not_full_.notify_all();
      return;
    }
    not_empty_.notify_one();
  }

  bool try_pop(std::vector<uint8_t>& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

  bool pop_wait(std::vector<uint8_t>& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!not_empty_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                             [&] { return !q_.empty() || !g_run.load(); }))
      return false;
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::deque<std::vector<uint8_t>> q_;
};

/** One RGBA frame for the file capturer queue (parallel JPEG workers use `seq` for filename order). */
struct CapturerFrameItem {
  uint32_t seq = 0;
  std::vector<uint8_t> rgba;
};

/** Capturer queue: raster **never blocks** when full — drops oldest RGBA so encode swap / JPEG backlog cannot stall the GPU path. */
class CapturerFrameQueue {
 public:
  explicit CapturerFrameQueue(int capacity) : cap_(std::max(1, capacity)) {}

  /** @return false if push failed (e.g. OOM); caller may need to undo `sess_pushed`. */
  bool push(CapturerFrameItem&& item, const char* producer_stage = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!g_run.load(std::memory_order_relaxed)) return false;
    uint32_t ndrop = 0;
    while ((int)q_.size() >= cap_) {
      if (q_.empty()) break;
      q_.pop_front();
      ++ndrop;
      g_capturer_frame_queue_drops.fetch_add(1, std::memory_order_relaxed);
    }
    if (ndrop)
      plotter_warn_queue_drop_oldest(PlotterQueueDropKind::CapturerFrame, "CapturerFrameQueue", producer_stage, ndrop);
    try {
      q_.push_back(std::move(item));
    } catch (const std::bad_alloc&) {
      fprintf(stderr, "resgrid_plotter: CapturerFrameQueue OOM; stopping pipeline\n");
      g_run.store(false, std::memory_order_relaxed);
      not_empty_.notify_all();
      return false;
    }
    not_empty_.notify_one();
    return true;
  }

  bool pop_wait(CapturerFrameItem& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!not_empty_.wait_for(lk, std::chrono::milliseconds(wait_ms), [&] {
          return !q_.empty() || !g_run.load(std::memory_order_relaxed) || wake_all_.load(std::memory_order_relaxed);
        }))
      return false;
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void notify_all_waiters() {
    wake_all_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lk(mu_);
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  void clear_wake_flag() { wake_all_.store(false, std::memory_order_release); }

  /** Drop captured RGBA backing store after workers joined; unblocks raster if queue was full. */
  void clear() {
    std::lock_guard<std::mutex> lk(mu_);
    q_.clear();
    not_full_.notify_all();
  }

  /** Frames waiting in the deque (excludes frames currently being JPEG-encoded after pop). */
  int size() {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)q_.size();
  }

  int capacity() const { return cap_; }

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::atomic<bool> wake_all_{false};
  std::deque<CapturerFrameItem> q_;
};

/** SHM-armed still capture to JPEG files; owned by `plotter_management`, fed from raster (parallel to `EncodeStageCtx` → video). */
struct CapturerCtx {
  const std::string* out_root_dir = nullptr;
  void* fps_base = nullptr;
  int frame_w = 0;
  int frame_h = 0;
  CapturerFrameQueue* frame_q = nullptr;

  std::mutex session_mu;
  std::vector<std::thread> file_workers;
  std::atomic<uint32_t> sess_pushed{0};
  std::atomic<uint32_t> sess_written{0};
  std::atomic<uint32_t> sess_tot{0};
  std::atomic<bool> sess_running{false};
  std::atomic<bool> sess_done{false};
  /** Items popped from the queue and currently in stbi_write_jpg (for drain-complete when queue drops lose frames). */
  std::atomic<int> jpeg_inflight{0};
  std::string sess_out_dir;
};

void capturer_maybe_write(CapturerCtx* c, const uint8_t* rgba, size_t nbytes);
/** Join JPEG worker pool (plotter shutdown or between idle cleanup). */
void capturer_join_workers(CapturerCtx* c);
/** Called from the stats thread: join finished JPEG workers when SHM is idle even if raster did not tick (`capturer_maybe_write`). */
void capturer_poll_idle_cleanup(CapturerCtx* c);

/** Context for the Raster stage thread (merge/acquire → CUDA + overlays → `RgbaFrameQueue`). */
struct RasterThreadCtx {
  int dm = 0;
  InternalFrameQueue* internal_q = nullptr;
  FullGridFifoQueue* full_grid_q = nullptr;
  AssembledSlotQueue* assembled_q = nullptr;
  GridFrameQueue* sidecar_q = nullptr;
  RgbaFrameQueue* rgba_out = nullptr;
  void* fps_base = nullptr;
  uint64_t idle_synth_fr = 0;
  int ncells = 0;
  int prb = 0;
  int sym = 0;
  int frame_w = 0;
  int frame_h = 0;
  int off_x = 0;
  int off_y = 0;
  int content_w = 0;
  int content_h = 0;
  float mag_scale = 1.f;
  float2* d_z = nullptr;
  float* d_mag = nullptr;
  uint8_t* d_chan = nullptr;
  /** Per-cell RNTI (slot metadata); same length as `d_chan`; zeros when no template. */
  uint32_t* d_cell_rnti = nullptr;
  uint8_t* d_rgba = nullptr;
  dim3 grid_mag{};
  dim3 block_mag{};
  dim3 grid2{};
  dim3 block2{};
  /** Non-null: async H2D / kernels / D2H on this stream (see `plotter_management.cu`). */
  cudaStream_t cuda_stream = nullptr;
  /** When set, armed file capturer samples each RGBA here (raster rate), before `rgba_out` push to encoder/video. */
  CapturerCtx* capturer = nullptr;
};

/** One grid to rasterize in pool mode; `seq` preserves assembly order at the encoder queue. */
struct RasterWorkItem {
  uint64_t seq = 0;
  GridFrame gf;
};

/** Bounded queue between single-thread acquirer and raster worker pool (no frame drops). */
class RasterWorkQueue {
 public:
  explicit RasterWorkQueue(int capacity);
  void push(RasterWorkItem&& frame, const char* producer_stage = nullptr);
  bool pop(RasterWorkItem& out, int wait_ms);
  bool try_pop(RasterWorkItem& out);

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::deque<RasterWorkItem> q_;
};

/** Merger: workers submit `pending[seq]`; one thread pushes `rgba_out` in `seq` order. */
struct RasterPoolCtrl {
  RgbaFrameQueue* rgba_out = nullptr;
  CapturerCtx* capturer = nullptr;
  std::mutex mu;
  std::condition_variable cv;
  std::unordered_map<uint64_t, std::vector<uint8_t>> pending;
  uint64_t next_merged_seq = 0;
  /** Monotonic: valid issued seq are `0 .. acquire_issued_end-1` once `acquire_done`. */
  uint64_t acquire_issued_end = 0;
  bool acquire_done = false;
};

struct RasterAcquireParams {
  RasterWorkQueue* work = nullptr;
  RasterPoolCtrl* pool = nullptr;
  int dm = 0;
  InternalFrameQueue* internal_q = nullptr;
  FullGridFifoQueue* full_grid_q = nullptr;
  AssembledSlotQueue* assembled_q = nullptr;
  GridFrameQueue* sidecar_q = nullptr;
  void* fps_base = nullptr; /* SHM pattern for synthetic idle frame in dm==3 */
  int ncells = 0;
  int prb = 0;
  int sym = 0;
};

/** Assembly stage: FIFO readers + slot assembler → `GridFrame` / assembled queues (one thread entry from management). */
struct AssemblyStageCtx {
  int dm = 0;
  InternalFrameQueue* internal_q = nullptr;
  void* fps_base = nullptr;
  int fps = 200;
  int ncells = 0;
  int prb = 0;
  int sym = 0;
  FullGridFifoQueue* full_grid_q = nullptr;
  int iq_fd_full_grid = -1;
  AssembledSlotQueue* slot_assembled_q = nullptr;
  const std::string* slot_fifo_path = nullptr;
  ChannelModeIqSidecarQueue* iq_sidecar_q = nullptr;
  int iq_fd_sidecar = -1;
};

/** Raster stage: single `raster_stage_main` or pool (supervisor joins children). */
struct RasterSupervisorCtx {
  int rt = 1;
  RasterThreadCtx* ctxs = nullptr;
  RasterWorkQueue* work_q = nullptr;
  RasterPoolCtrl* pool = nullptr;
  RasterAcquireParams* ap = nullptr;
  size_t rgba_bytes = 0;
};

/** CUDA IQ SHM size; keep in sync with `backend/app/main.py` `_IQ_CUDA_SHM_BYTES`. */
inline constexpr size_t kIqCudaShmBytes = 384;
/** File capturer SHM: total @88, inflight @92, disk @96, phase @100, path @104; tail @360 queue depth/cap/JPEG threads; stop @372. */
inline constexpr size_t kShmCapturerTotalOff = 88;
inline constexpr size_t kShmCapturerInflightOff = 92;
inline constexpr size_t kShmCapturerWrittenOff = 96;
inline constexpr size_t kShmCapturerPhaseOff = 100;
inline constexpr size_t kShmCapturerPathOff = 104;
inline constexpr size_t kShmCapturerPathCap = 256;
inline constexpr size_t kShmCapturerQueueDepthOff = 360;
inline constexpr size_t kShmCapturerQueueCapOff = 364;
inline constexpr size_t kShmCapturerJpegThreadsOff = 368;
/** Backend sets to 1 to request early end; plotter clears and caps **total** to **sess_pushed** (disk flush continues). */
inline constexpr size_t kShmCapturerStopOff = 372;
/** 0 idle, 1 recording (buffering RGBA), 2 writing JPEGs in background. */
inline constexpr uint32_t kCapturerPhaseIdle = 0;
inline constexpr uint32_t kCapturerPhaseRecording = 1;
inline constexpr uint32_t kCapturerPhaseWriting = 2;

/** Encode stage (video): `RgbaFrameQueue` → FFmpeg stdin → NVENC/RTSP. Parallel file path: `CapturerCtx` + `capturer_maybe_write` from raster. */
struct EncodeStageCtx {
  RgbaFrameQueue* rgba_q = nullptr;
  /** Write end of pipe to FFmpeg stdin (raw `write(2)` only — do not mix with stdio on this fd). */
  int ff_pipe_fd = -1;
  EncodePolicy enc_pol = EncodePolicy::Last;
  int dm = 0;
  int encoder_fps = 24;
  int fps = 200;
  void* fps_base = nullptr;
  size_t rgba_npixels = 0;
  size_t rgba_bytes = 0;
  std::vector<float> mean_acc_r;
  std::vector<float> mean_acc_g;
  std::vector<float> mean_acc_b;
  std::vector<uint32_t> mean_acc_a;
  uint64_t rgba_mean_count = 0;
};

#pragma pack(push, 1)
struct SlotMsgHdr {
  uint32_t magic;
  uint16_t version;
  uint16_t flags;
  uint32_t slot_id;
  uint16_t sym_start;
  uint16_t sym_count;
  uint16_t prb_start;
  uint16_t prb_count;
  uint32_t rnti;
  uint8_t channel;
  uint8_t reserved0;
  uint16_t reserved1;
  uint32_t n_complex;
  uint32_t reserved2;
  uint32_t reserved3;
};
#pragma pack(pop)

static_assert(sizeof(SlotMsgHdr) == 40, "SlotMsgHdr size");

/** One rectangular allocation inside a slot (v2 compound messages; also maps from v1 header). */
#pragma pack(push, 1)
struct SlotRegionDesc {
  uint16_t sym_start;
  uint16_t sym_count;
  uint16_t prb_start;
  uint16_t prb_count;
  uint32_t rnti;
  uint8_t channel;
  uint8_t reserved[3];
};
#pragma pack(pop)

static_assert(sizeof(SlotRegionDesc) == 16, "SlotRegionDesc size");


inline void fps_shm_store(void* base, size_t off, double v) {
  if (!base) return;
  std::memcpy(static_cast<char*>(base) + off, &v, sizeof(v));
}

inline void fps_shm_store_u64(void* base, size_t off, uint64_t v) {
  if (!base) return;
  std::memcpy(static_cast<char*>(base) + off, &v, sizeof(v));
}

inline int fps_shm_command_hz(const void* base, int fallback_hz) {
  if (!base) return std::max(1, fallback_hz);
  double cmd;
  std::memcpy(&cmd, static_cast<const char*>(base) + 24, sizeof(cmd));
  if (!std::isfinite(cmd) || cmd < 1.0) return std::max(1, fallback_hz);
  int h = (int)std::lrint(cmd);
  if (h > 600) h = 600;
  return std::max(1, h);
}

inline constexpr uint32_t kIqPatternMax = 4;

inline uint32_t fps_shm_pattern_u32(const void* base) {
  if (!base) return 0;
  uint32_t p;
  std::memcpy(&p, static_cast<const char*>(base) + 16, sizeof(p));
  if (p > kIqPatternMax) return 0;
  return p;
}

void run_internal_synthetic_producer_loop(InternalFrameQueue* q, void* fps_base, int fps_default, int ncells, int prb,
                                            int sym);
/** Open IQ named pipe read end (assembly); used from `main` before starting reader threads. */
int open_iq_fifo_read_end(const char* path, const char* wait_label);
void external_grid_frame_reader_main(int fd, int ncells, GridFrameQueue* outq, const char* producer_stage);
void slot_message_assembler_main(const std::string& path, int prb_dim, int sym_dim, int ncells, AssembledSlotQueue* outq);
void raster_stage_main(RasterThreadCtx* ctx);
void raster_acquirer_main(RasterAcquireParams* p);
void raster_worker_main(RasterThreadCtx* ctx, RasterWorkQueue* wq, RasterPoolCtrl* pool, size_t rgba_bytes);
void raster_merger_main(RasterPoolCtrl* pool);
void assembly_stage_main(AssemblyStageCtx* ctx);
void raster_stage_supervisor_main(RasterSupervisorCtx* ctx);
void encode_stage_main(EncodeStageCtx* ctx);
/** @return Write end of pipe to child FFmpeg stdin, or **-1** on failure. */
int spawn_ffmpeg_nvenc_rtsp(const char* url, int w, int h, int encoder_fps, const char* preset, pid_t* pid_out,
                            EncodePolicy policy, int pipeline_fps);
bool write_rgba_to_ffmpeg(EncodeStageCtx* ec, const uint8_t* rgba, size_t nbytes);
void fill_synthetic_iq(std::vector<float2>& out, std::vector<uint8_t>& chan, std::vector<uint32_t>& cell_rnti, int prb,
                       int sym, uint64_t frame, uint32_t pattern);
