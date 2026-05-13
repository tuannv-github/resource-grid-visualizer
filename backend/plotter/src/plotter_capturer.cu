/*
 * Raster-time still capture: `CapturerFrameQueue` drops oldest RGBA when full so raster never blocks (e.g. Swap / slow JPEG).
 * CPU JPEG workers write `frame_*.jpg`; parallel to encode (RGBA → FFmpeg → RTSP).
 */
#include "plotter_common.cuh"
#include "plotter_capturer.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <new>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <malloc.h>
#endif

static uint32_t capturer_jpeg_worker_count() {
  if (const char* e = std::getenv("RESGRID_CAPTURE_JPEG_THREADS")) {
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end != e && v >= 1 && v <= 64) return static_cast<uint32_t>(v);
  }
  unsigned hc = std::thread::hardware_concurrency();
  uint32_t n = hc == 0 ? 4u : static_cast<uint32_t>(hc);
  if (n > 16u) n = 16u;
  return n;
}

static bool capturer_output_dir_allowed(const std::string& base, const char* out_dir) {
  if (base.empty() || !out_dir || !out_dir[0]) return false;
  char* rp_b = realpath(base.c_str(), nullptr);
  if (!rp_b) return false;
  char* rp_o = realpath(out_dir, nullptr);
  if (!rp_o) {
    free(rp_b);
    return false;
  }
  const size_t lb = std::strlen(rp_b);
  const bool under =
      std::strncmp(rp_o, rp_b, lb) == 0 && (rp_o[lb] == '\0' || rp_o[lb] == '/');
  free(rp_b);
  free(rp_o);
  return under;
}

static void shm_abort_capturer(void* fps_base) {
  if (!fps_base) return;
  char* shm = static_cast<char*>(fps_base);
  const uint32_t z = 0;
  std::memcpy(shm + kShmCapturerTotalOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerStopOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerPhaseOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerInflightOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerWrittenOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerQueueDepthOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerQueueCapOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerJpegThreadsOff, &z, sizeof(z));
}

void CapturerStage::capturer_configure(const std::string* out_root_dir, void* fps_base, int frame_w, int frame_h,
                              CapturerFrameQueue* frame_q) {
  out_root_dir_ = out_root_dir;
  fps_base_ = fps_base;
  frame_w_ = frame_w;
  frame_h_ = frame_h;
  frame_q_ = frame_q;
}

void CapturerStage::publish_queue_depth() {
  if (!fps_base_ || !frame_q_) return;
  const uint32_t d = (uint32_t)frame_q_->size();
  std::memcpy(static_cast<char*>(fps_base_) + kShmCapturerQueueDepthOff, &d, sizeof(d));
}

void CapturerStage::mark_session_complete_shm() {
  if (!fps_base_) return;
  char* shm = static_cast<char*>(fps_base_);
  const uint32_t z = 0;
  std::memcpy(shm + kShmCapturerPhaseOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerInflightOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerQueueDepthOff, &z, sizeof(z));
  sess_done_.store(true, std::memory_order_release);
  sess_running_.store(false, std::memory_order_release);
  if (frame_q_) frame_q_->notify_all_waiters();
#if defined(__linux__)
  malloc_trim(0);
#endif
}

bool CapturerStage::drain_complete(char* shm) {
  uint32_t ph = 0;
  std::memcpy(&ph, shm + kShmCapturerPhaseOff, sizeof(ph));
  if (ph != kCapturerPhaseWriting) return false;
  const uint32_t pushed = sess_pushed_.load(std::memory_order_acquire);
  const uint32_t ttot = sess_tot_.load(std::memory_order_acquire);
  if (ttot == 0 || pushed < ttot) return false;
  if (frame_q_->size() != 0) return false;
  if (jpeg_inflight_.load(std::memory_order_acquire) != 0) return false;
  mark_session_complete_shm();
  return true;
}

void CapturerStage::jpeg_worker_main(CapturerStage* self) {
  char* shm = static_cast<char*>(self->fps_base_);
  const int fw = self->frame_w_;
  const int fh = self->frame_h_;
  thread_local static std::vector<uint8_t> work_rgba;

  for (;;) {
    if (!g_run.load(std::memory_order_relaxed)) break;
    if (self->sess_done_.load(std::memory_order_acquire)) break;

    CapturerFrameItem it;
    if (!self->frame_q_->pop_wait(it, 150)) {
      const uint32_t w = self->sess_written_.load(std::memory_order_relaxed);
      const uint32_t ttot = self->sess_tot_.load(std::memory_order_relaxed);
      if (ttot > 0 && w >= ttot) {
        self->mark_session_complete_shm();
        break;
      }
      if (self->drain_complete(shm)) break;
      continue;
    }

    self->publish_queue_depth();

    if (it.rgba.size() != (size_t)fw * (size_t)fh * 4u) {
      fprintf(stderr, "resgrid_plotter: capturer frame size mismatch (seq=%u)\n", it.seq);
      continue;
    }

    self->jpeg_inflight_.fetch_add(1, std::memory_order_relaxed);
    work_rgba.swap(it.rgba);
    const uint32_t frame_seq = it.seq;

    std::string out_dir;
    {
      std::lock_guard<std::mutex> lk(self->session_mu_);
      out_dir = self->sess_out_dir_;
    }
    char fn[PATH_MAX];
    const int n = std::snprintf(fn, sizeof(fn), "%s/frame_%06u.jpg", out_dir.c_str(), frame_seq);
    if (n < 0 || (size_t)n >= sizeof(fn)) {
      fprintf(stderr, "resgrid_plotter: capturer jpg path too long\n");
      work_rgba.clear();
      self->jpeg_inflight_.fetch_sub(1, std::memory_order_relaxed);
      continue;
    }
    if (!stbi_write_jpg(fn, fw, fh, 4, work_rgba.data(), 92)) {
      fprintf(stderr, "resgrid_plotter: stbi_write_jpg failed %s\n", fn);
      shm_abort_capturer(self->fps_base_);
      self->jpeg_inflight_.fetch_sub(1, std::memory_order_relaxed);
      self->sess_done_.store(true, std::memory_order_release);
      self->sess_running_.store(false, std::memory_order_release);
      self->frame_q_->notify_all_waiters();
      work_rgba.clear();
      break;
    }

    work_rgba.clear();

    const uint32_t w = self->sess_written_.fetch_add(1, std::memory_order_acq_rel) + 1u;
    self->jpeg_inflight_.fetch_sub(1, std::memory_order_relaxed);
    const uint32_t qdep = (uint32_t)self->frame_q_->size();
    const int ji = self->jpeg_inflight_.load(std::memory_order_acquire);
    const uint32_t shm_infl = qdep + (uint32_t)std::max(0, ji);
    std::memcpy(shm + kShmCapturerWrittenOff, &w, sizeof(w));
    std::memcpy(shm + kShmCapturerInflightOff, &shm_infl, sizeof(shm_infl));

    const uint32_t tot = self->sess_tot_.load(std::memory_order_acquire);
    if (w == tot && tot > 0) {
      self->mark_session_complete_shm();
      break;
    }
  }
}

void CapturerStage::join_workers_internal() {
  sess_done_.store(true, std::memory_order_release);
  sess_running_.store(false, std::memory_order_release);
  if (frame_q_) frame_q_->notify_all_waiters();
  std::lock_guard<std::mutex> lk(session_mu_);
  for (auto& t : file_workers_) {
    if (t.joinable()) t.join();
  }
  file_workers_.clear();
  if (frame_q_) frame_q_->clear_wake_flag();
  sess_out_dir_.clear();
  sess_out_dir_.shrink_to_fit();
  sess_pushed_.store(0, std::memory_order_release);
  sess_written_.store(0, std::memory_order_release);
  sess_tot_.store(0, std::memory_order_release);
  sess_done_.store(false, std::memory_order_release);
  jpeg_inflight_.store(0, std::memory_order_release);
  if (frame_q_) {
    frame_q_->clear();
    publish_queue_depth();
  }
#if defined(__linux__)
  malloc_trim(0);
#endif
}

void CapturerStage::join() { join_workers_internal(); }

void CapturerStage::poll_idle_cleanup() {
  if (!fps_base_ || !out_root_dir_ || out_root_dir_->empty() || !frame_q_) return;
  char* shm = static_cast<char*>(fps_base_);
  uint32_t phase = 0;
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));
  if (phase != kCapturerPhaseIdle) return;
  bool need_join = false;
  {
    std::lock_guard<std::mutex> lk(session_mu_);
    need_join = !file_workers_.empty();
  }
  if (need_join) join_workers_internal();
}

void CapturerStage::maybe_write(const uint8_t* rgba, size_t nbytes) {
  if (!out_root_dir_ || out_root_dir_->empty() || !frame_q_) return;
  void* fps = fps_base_;
  if (!fps) return;
  if (nbytes != (size_t)frame_w_ * (size_t)frame_h_ * 4u) return;
  char* shm = static_cast<char*>(fps);

  uint32_t tot = 0;
  uint32_t phase = 0;
  std::memcpy(&tot, shm + kShmCapturerTotalOff, sizeof(tot));
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));

  if (phase == kCapturerPhaseIdle) {
    bool need_join = false;
    {
      std::lock_guard<std::mutex> lk(session_mu_);
      need_join = !file_workers_.empty();
    }
    if (need_join) join_workers_internal();
    return;
  }
  if (phase == kCapturerPhaseRecording && tot > 0) {
    uint32_t stop_req = 0;
    std::memcpy(&stop_req, shm + kShmCapturerStopOff, sizeof(stop_req));
    if (stop_req != 0) {
      const uint32_t z = 0;
      std::memcpy(shm + kShmCapturerStopOff, &z, sizeof(z));
      const uint32_t cap = sess_pushed_.load(std::memory_order_acquire);
      sess_tot_.store(cap, std::memory_order_release);
      std::memcpy(shm + kShmCapturerTotalOff, &cap, sizeof(cap));
      tot = cap;
      if (cap == 0) {
        join_workers_internal();
        shm_abort_capturer(fps);
        return;
      }
      const uint32_t pushed = sess_pushed_.load(std::memory_order_acquire);
      if (pushed >= cap) {
        const uint32_t p_write = kCapturerPhaseWriting;
        std::memcpy(shm + kShmCapturerPhaseOff, &p_write, sizeof(p_write));
      }
    }
  }
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));
  if (phase != kCapturerPhaseRecording || tot == 0) return;

  {
    std::unique_lock<std::mutex> lk(session_mu_);
    if (!file_workers_.empty() && !sess_running_.load(std::memory_order_acquire)) {
      lk.unlock();
      join_workers_internal();
      lk.lock();
    }
    if (file_workers_.empty()) {
      char pathbuf[kShmCapturerPathCap];
      std::memcpy(pathbuf, shm + kShmCapturerPathOff, kShmCapturerPathCap);
      pathbuf[kShmCapturerPathCap - 1] = '\0';
      if (!pathbuf[0] || !capturer_output_dir_allowed(*out_root_dir_, pathbuf)) {
        fprintf(stderr, "resgrid_plotter: capturer output dir invalid; aborting\n");
        shm_abort_capturer(fps);
        return;
      }
      sess_out_dir_.assign(pathbuf);
      sess_tot_.store(tot, std::memory_order_release);
      sess_pushed_.store(0, std::memory_order_release);
      sess_written_.store(0, std::memory_order_release);
      sess_done_.store(false, std::memory_order_release);
      jpeg_inflight_.store(0, std::memory_order_release);
      sess_running_.store(true, std::memory_order_release);
      const uint32_t dz = 0;
      std::memcpy(shm + kShmCapturerWrittenOff, &dz, sizeof(dz));

      const uint32_t nworkers = capturer_jpeg_worker_count();
      for (uint32_t i = 0; i < nworkers; ++i) file_workers_.emplace_back(jpeg_worker_main, this);
      const uint32_t cap_u = (uint32_t)std::max(1, frame_q_->capacity());
      std::memcpy(shm + kShmCapturerQueueCapOff, &cap_u, sizeof(cap_u));
      std::memcpy(shm + kShmCapturerJpegThreadsOff, &nworkers, sizeof(nworkers));
      publish_queue_depth();
    }
  }

  if (sess_pushed_.load(std::memory_order_acquire) >= tot) return;

  CapturerFrameItem it;
  it.seq = 0;
  try {
    it.rgba.assign(rgba, rgba + nbytes);
  } catch (const std::bad_alloc&) {
    fprintf(stderr, "resgrid_plotter: capturer OOM (RGBA copy); aborting capture and pipeline\n");
    shm_abort_capturer(fps);
    join_workers_internal();
    g_run.store(false, std::memory_order_relaxed);
    return;
  }
  const uint32_t seq = sess_pushed_.fetch_add(1, std::memory_order_acq_rel);
  it.seq = seq;
  if (!frame_q_->push(std::move(it), "Raster (capturer)")) {
    sess_pushed_.fetch_sub(1, std::memory_order_acq_rel);
    shm_abort_capturer(fps);
    join_workers_internal();
    return;
  }

  const uint32_t pushed = sess_pushed_.load(std::memory_order_acquire);
  const uint32_t qdep = (uint32_t)frame_q_->size();
  const int ji = jpeg_inflight_.load(std::memory_order_acquire);
  const uint32_t inflight = qdep + (uint32_t)std::max(0, ji);
  std::memcpy(shm + kShmCapturerInflightOff, &inflight, sizeof(inflight));
  publish_queue_depth();

  if (pushed >= tot) {
    const uint32_t p_write = kCapturerPhaseWriting;
    std::memcpy(shm + kShmCapturerPhaseOff, &p_write, sizeof(p_write));
  }
}
