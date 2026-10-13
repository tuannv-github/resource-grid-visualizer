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
#include <memory>
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
 * (Assembler → one `GridFrame` to downstream; Raster → one RGBA frame; Encoder → one full NVENC stdin write).
 * The pipeline stats thread samples these at ~1 s intervals and publishes **Δcount / Δt** as Hz (see `plotter_management.cu`).
 */
extern std::atomic<uint64_t> g_stage_output_count_assembler;
extern std::atomic<uint64_t> g_stage_output_count_raster;
extern std::atomic<uint64_t> g_stage_output_count_encoder;
extern std::atomic<uint64_t> g_slot_completed_drops;
extern std::atomic<uint64_t> g_grid_frame_queue_drops;
extern std::atomic<uint64_t> g_rgba_frame_queue_drops;
extern std::atomic<uint64_t> g_raster_work_queue_drops;
extern std::atomic<uint64_t> g_capturer_frame_queue_drops;
extern std::atomic<uint64_t> g_assembler_input_msgs;
/** 1 while the IQ reader is replaying the last real slot because no newer slot arrived. */
extern std::atomic<uint32_t> g_input_held;
/** CUDA control SHM mapped by the plotter; null before launch and after shutdown. */
extern void* g_plotter_fps_shm;
/** uint32 in the CUDA control SHM: 1 = replaying the last slot at 1 Hz. Bytes 376…379 (after capturer stop @372). */
inline constexpr size_t kShmInputHeldOff = 376;

inline void plotter_store_input_held(bool held) {
  const uint32_t v = held ? 1u : 0u;
  g_input_held.store(v, std::memory_order_relaxed);
  void* const base = g_plotter_fps_shm;
  if (!base) return;
  std::memcpy(static_cast<char*>(base) + kShmInputHeldOff, &v, sizeof(v));
}

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
  /** Set when the ZeroMQ header carried ``sfn=`` and ``slot=`` (slot index within that SFN). */
  bool has_timing = false;
  uint16_t sfn = 0;
  uint16_t slot = 0;
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
/** Pipeline 2 (full-grid mq): one `--mq-resource-grid` logical frame = one complete resource-grid I/Q raster. */
using FullGridFifoQueue = GridFrameQueue;
/** Pipeline 3: slot-message assembler pushes one fully assembled grid (from slot allocations) here. */
using AssembledSlotQueue = SlotCompletedQueue;
/** Pipeline 3 optional: `--mq-resource-grid` full-grid sidecar while FAPI messages supply channel metadata / patches. */
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

#include "plotter_capturer.cuh"
#include "plotter_raster.cuh"
#include "plotter_assembler.cuh"

/** CUDA IQ SHM size; keep in sync with `backend/app/main.py` `_IQ_CUDA_SHM_BYTES`. */
inline constexpr size_t kIqCudaShmBytes = 384;

/** Encoder stage (video): `RgbaFrameQueue` → FFmpeg stdin → NVENC/RTSP. Parallel file path: `CapturerStage::maybe_write` from raster. */
struct EncoderStageCtx {
  RgbaFrameQueue* rgba_q = nullptr;
  /** Write end of pipe to FFmpeg stdin (raw `write(2)` only — do not mix with stdio on this fd). */
  int ff_pipe_fd = -1;
  EncodePolicy enc_pol = EncodePolicy::Last;
  /** True when `--mq-fapi` is active: encoder does not inject filler RGBA on idle raster ticks. */
  bool fapi_slot_path = false;
  /** True when standalone resource-grid FIFO uses slot-IQ framing (sparse cadence like FAPI). */
  bool resource_grid_slot_iq_stream = false;
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

#include "plotter_encoder.cuh"

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

/** Standalone ``--mq-resource-grid`` slot-IQ stream (little-endian, packed). One message = one rectangular
 *  patch of complex samples for a ``slot_id``; multiple messages may accumulate before ``slot_id`` advances.
 *  Python: ``struct.pack("<IIHHHHI", magic, slot_id, sym_start, sym_count, prb_start, prb_count, n_complex)``. */
#pragma pack(push, 1)
struct ResourceGridSlotIqHdr {
  uint32_t magic;
  uint32_t slot_id;
  uint16_t sym_start;
  uint16_t sym_count;
  uint16_t prb_start;
  uint16_t prb_count;
  uint32_t n_complex;
};
#pragma pack(pop)

static_assert(sizeof(ResourceGridSlotIqHdr) == 20, "ResourceGridSlotIqHdr size");

/** Magic for ``ResourceGridSlotIqHdr`` (ASCII ``RGIS``). */
inline constexpr uint32_t kResourceGridSlotIqMagic = 0x52474953u;

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

void fill_synthetic_iq(std::vector<float2>& out, std::vector<uint8_t>& chan, std::vector<uint32_t>& cell_rnti, int prb,
                       int sym, uint64_t frame, uint32_t pattern);
