#pragma once
/** File capturer: RGBA queue, SHM offsets, JPEG workers. Include only after `PlotterQueueDropKind` and `g_run` in
 *  `plotter_common.cuh`. */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

/** Raster-time JPEG capture (parallel to encode); SHM-armed stills, wired from `plotter_run_launch`. */
class CapturerStage {
 public:
  void capturer_configure(const std::string* out_root_dir, void* fps_base, int frame_w, int frame_h,
                          CapturerFrameQueue* frame_q);

  void maybe_write(const uint8_t* rgba, size_t nbytes);
  void join();
  void poll_idle_cleanup();

 private:
  static void jpeg_worker_main(CapturerStage* self);

  void publish_queue_depth();
  void mark_session_complete_shm();
  bool drain_complete(char* shm);
  void join_workers_internal();

  const std::string* out_root_dir_ = nullptr;
  void* fps_base_ = nullptr;
  int frame_w_ = 0;
  int frame_h_ = 0;
  CapturerFrameQueue* frame_q_ = nullptr;

  std::mutex session_mu_;
  std::vector<std::thread> file_workers_;
  std::atomic<uint32_t> sess_pushed_{0};
  std::atomic<uint32_t> sess_written_{0};
  std::atomic<uint32_t> sess_tot_{0};
  std::atomic<bool> sess_running_{false};
  std::atomic<bool> sess_done_{false};
  std::atomic<int> jpeg_inflight_{0};
  std::string sess_out_dir_;
};

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
