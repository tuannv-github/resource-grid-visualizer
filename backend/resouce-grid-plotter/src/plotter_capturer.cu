/*
 * Raster-time still capture: `CapturerFrameQueue` drops oldest RGBA when full so raster never blocks (e.g. Swap / slow JPEG).
 * CPU JPEG workers write `frame_*.jpg`; parallel to encode (RGBA → FFmpeg → RTSP).
 */
#include "plotter_common.cuh"

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

static void capturer_publish_queue_depth(CapturerCtx* c) {
  if (!c || !c->fps_base || !c->frame_q) return;
  const uint32_t d = (uint32_t)c->frame_q->size();
  std::memcpy(static_cast<char*>(c->fps_base) + kShmCapturerQueueDepthOff, &d, sizeof(d));
}

/** One worker may finish via pop timeout instead of the write path; both must mark SHM idle or phase stays "saving". */
static void capturer_mark_session_complete_shm(CapturerCtx* c) {
  if (!c || !c->fps_base) return;
  char* shm = static_cast<char*>(c->fps_base);
  const uint32_t z = 0;
  std::memcpy(shm + kShmCapturerPhaseOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerInflightOff, &z, sizeof(z));
  std::memcpy(shm + kShmCapturerQueueDepthOff, &z, sizeof(z));
  c->sess_done.store(true, std::memory_order_release);
  c->sess_running.store(false, std::memory_order_release);
  if (c->frame_q) c->frame_q->notify_all_waiters();
#if defined(__linux__)
  malloc_trim(0);
#endif
}

/** After queue drops, `written` may never reach `sess_tot`; finish when raster is done, deque empty, encoders idle. */
static bool capturer_drain_complete(CapturerCtx* c, char* shm) {
  uint32_t ph = 0;
  std::memcpy(&ph, shm + kShmCapturerPhaseOff, sizeof(ph));
  if (ph != kCapturerPhaseWriting) return false;
  const uint32_t pushed = c->sess_pushed.load(std::memory_order_acquire);
  const uint32_t ttot = c->sess_tot.load(std::memory_order_acquire);
  if (ttot == 0 || pushed < ttot) return false;
  if (c->frame_q->size() != 0) return false;
  if (c->jpeg_inflight.load(std::memory_order_acquire) != 0) return false;
  capturer_mark_session_complete_shm(c);
  return true;
}

static void capturer_file_worker(CapturerCtx* c) {
  char* shm = static_cast<char*>(c->fps_base);
  const int fw = c->frame_w;
  const int fh = c->frame_h;
  thread_local static std::vector<uint8_t> work_rgba;

  for (;;) {
    if (!g_run.load(std::memory_order_relaxed)) break;
    if (c->sess_done.load(std::memory_order_acquire)) break;

    CapturerFrameItem it;
    if (!c->frame_q->pop_wait(it, 150)) {
      const uint32_t w = c->sess_written.load(std::memory_order_relaxed);
      const uint32_t ttot = c->sess_tot.load(std::memory_order_relaxed);
      if (ttot > 0 && w >= ttot) {
        capturer_mark_session_complete_shm(c);
        break;
      }
      if (capturer_drain_complete(c, shm)) break;
      continue;
    }

    capturer_publish_queue_depth(c);

    if (it.rgba.size() != (size_t)fw * (size_t)fh * 4u) {
      fprintf(stderr, "resgrid_plotter: capturer frame size mismatch (seq=%u)\n", it.seq);
      continue;
    }

    c->jpeg_inflight.fetch_add(1, std::memory_order_relaxed);
    work_rgba.swap(it.rgba);
    const uint32_t frame_seq = it.seq;

    std::string out_dir;
    {
      std::lock_guard<std::mutex> lk(c->session_mu);
      out_dir = c->sess_out_dir;
    }
    char fn[PATH_MAX];
    const int n = std::snprintf(fn, sizeof(fn), "%s/frame_%06u.jpg", out_dir.c_str(), frame_seq);
    if (n < 0 || (size_t)n >= sizeof(fn)) {
      fprintf(stderr, "resgrid_plotter: capturer jpg path too long\n");
      work_rgba.clear();
      c->jpeg_inflight.fetch_sub(1, std::memory_order_relaxed);
      continue;
    }
    if (!stbi_write_jpg(fn, fw, fh, 4, work_rgba.data(), 92)) {
      fprintf(stderr, "resgrid_plotter: stbi_write_jpg failed %s\n", fn);
      shm_abort_capturer(c->fps_base);
      c->jpeg_inflight.fetch_sub(1, std::memory_order_relaxed);
      c->sess_done.store(true, std::memory_order_release);
      c->sess_running.store(false, std::memory_order_release);
      c->frame_q->notify_all_waiters();
      work_rgba.clear();
      break;
    }

    work_rgba.clear();

    const uint32_t w = c->sess_written.fetch_add(1, std::memory_order_acq_rel) + 1u;
    c->jpeg_inflight.fetch_sub(1, std::memory_order_relaxed);
    const uint32_t qdep = (uint32_t)c->frame_q->size();
    const int ji = c->jpeg_inflight.load(std::memory_order_acquire);
    const uint32_t shm_infl = qdep + (uint32_t)std::max(0, ji);
    std::memcpy(shm + kShmCapturerWrittenOff, &w, sizeof(w));
    std::memcpy(shm + kShmCapturerInflightOff, &shm_infl, sizeof(shm_infl));

    const uint32_t tot = c->sess_tot.load(std::memory_order_acquire);
    if (w == tot && tot > 0) {
      capturer_mark_session_complete_shm(c);
      break;
    }
  }
}

void capturer_join_workers(CapturerCtx* c) {
  if (!c) return;
  c->sess_done.store(true, std::memory_order_release);
  c->sess_running.store(false, std::memory_order_release);
  if (c->frame_q) c->frame_q->notify_all_waiters();
  std::lock_guard<std::mutex> lk(c->session_mu);
  for (auto& t : c->file_workers) {
    if (t.joinable()) t.join();
  }
  c->file_workers.clear();
  if (c->frame_q) c->frame_q->clear_wake_flag();
  c->sess_out_dir.clear();
  c->sess_out_dir.shrink_to_fit();
  c->sess_pushed.store(0, std::memory_order_release);
  c->sess_written.store(0, std::memory_order_release);
  c->sess_tot.store(0, std::memory_order_release);
  c->sess_done.store(false, std::memory_order_release);
  c->jpeg_inflight.store(0, std::memory_order_release);
  if (c->frame_q) {
    c->frame_q->clear();
    capturer_publish_queue_depth(c);
  }
#if defined(__linux__)
  malloc_trim(0);
#endif
}

void capturer_poll_idle_cleanup(CapturerCtx* c) {
  if (!c || !c->fps_base || !c->out_root_dir || c->out_root_dir->empty() || !c->frame_q) return;
  char* shm = static_cast<char*>(c->fps_base);
  uint32_t phase = 0;
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));
  if (phase != kCapturerPhaseIdle) return;
  bool need_join = false;
  {
    std::lock_guard<std::mutex> lk(c->session_mu);
    need_join = !c->file_workers.empty();
  }
  if (need_join) capturer_join_workers(c);
}

void capturer_maybe_write(CapturerCtx* c, const uint8_t* rgba, size_t nbytes) {
  if (!c || !c->out_root_dir || c->out_root_dir->empty() || !c->frame_q) return;
  void* fps = c->fps_base;
  if (!fps) return;
  if (nbytes != (size_t)c->frame_w * (size_t)c->frame_h * 4u) return;
  char* shm = static_cast<char*>(fps);

  uint32_t tot = 0;
  uint32_t phase = 0;
  std::memcpy(&tot, shm + kShmCapturerTotalOff, sizeof(tot));
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));

  /* Idle: capturer context and queue exist whenever --capturer-dir is set, but we do not enqueue.
   * Join prior session threads only if joinable threads remain — not on every raster frame. */
  if (phase == kCapturerPhaseIdle) {
    bool need_join = false;
    {
      std::lock_guard<std::mutex> lk(c->session_mu);
      need_join = !c->file_workers.empty();
    }
    if (need_join) capturer_join_workers(c);
    return;
  }
  /* User stop: cap session at frames already accepted (no further raster enqueue). */
  if (phase == kCapturerPhaseRecording && tot > 0) {
    uint32_t stop_req = 0;
    std::memcpy(&stop_req, shm + kShmCapturerStopOff, sizeof(stop_req));
    if (stop_req != 0) {
      const uint32_t z = 0;
      std::memcpy(shm + kShmCapturerStopOff, &z, sizeof(z));
      const uint32_t cap = c->sess_pushed.load(std::memory_order_acquire);
      c->sess_tot.store(cap, std::memory_order_release);
      std::memcpy(shm + kShmCapturerTotalOff, &cap, sizeof(cap));
      tot = cap;
      if (cap == 0) {
        capturer_join_workers(c);
        shm_abort_capturer(fps);
        return;
      }
      const uint32_t pushed = c->sess_pushed.load(std::memory_order_acquire);
      if (pushed >= cap) {
        const uint32_t p_write = kCapturerPhaseWriting;
        std::memcpy(shm + kShmCapturerPhaseOff, &p_write, sizeof(p_write));
      }
    }
  }
  std::memcpy(&phase, shm + kShmCapturerPhaseOff, sizeof(phase));
  /* Writing: last frames are flushing to disk; raster does not enqueue. */
  if (phase != kCapturerPhaseRecording || tot == 0) return;

  {
    std::unique_lock<std::mutex> lk(c->session_mu);
    /* New arm can arrive before any idle raster frame joined exited workers; clear stale joinables. */
    if (!c->file_workers.empty() && !c->sess_running.load(std::memory_order_acquire)) {
      lk.unlock();
      capturer_join_workers(c);
      lk.lock();
    }
    if (c->file_workers.empty()) {
      char pathbuf[kShmCapturerPathCap];
      std::memcpy(pathbuf, shm + kShmCapturerPathOff, kShmCapturerPathCap);
      pathbuf[kShmCapturerPathCap - 1] = '\0';
      if (!pathbuf[0] || !capturer_output_dir_allowed(*c->out_root_dir, pathbuf)) {
        fprintf(stderr, "resgrid_plotter: capturer output dir invalid; aborting\n");
        shm_abort_capturer(fps);
        return;
      }
      c->sess_out_dir.assign(pathbuf);
      c->sess_tot.store(tot, std::memory_order_release);
      c->sess_pushed.store(0, std::memory_order_release);
      c->sess_written.store(0, std::memory_order_release);
      c->sess_done.store(false, std::memory_order_release);
      c->jpeg_inflight.store(0, std::memory_order_release);
      c->sess_running.store(true, std::memory_order_release);
      const uint32_t dz = 0;
      std::memcpy(shm + kShmCapturerWrittenOff, &dz, sizeof(dz));

      const uint32_t nworkers = capturer_jpeg_worker_count();
      for (uint32_t i = 0; i < nworkers; ++i) c->file_workers.emplace_back(capturer_file_worker, c);
      const uint32_t cap_u = (uint32_t)std::max(1, c->frame_q->capacity());
      std::memcpy(shm + kShmCapturerQueueCapOff, &cap_u, sizeof(cap_u));
      std::memcpy(shm + kShmCapturerJpegThreadsOff, &nworkers, sizeof(nworkers));
      capturer_publish_queue_depth(c);
    }
  }

  if (c->sess_pushed.load(std::memory_order_acquire) >= tot) return;

  CapturerFrameItem it;
  it.seq = 0;
  try {
    it.rgba.assign(rgba, rgba + nbytes);
  } catch (const std::bad_alloc&) {
    fprintf(stderr, "resgrid_plotter: capturer OOM (RGBA copy); aborting capture and pipeline\n");
    shm_abort_capturer(fps);
    capturer_join_workers(c);
    g_run.store(false, std::memory_order_relaxed);
    return;
  }
  const uint32_t seq = c->sess_pushed.fetch_add(1, std::memory_order_acq_rel);
  it.seq = seq;
  if (!c->frame_q->push(std::move(it), "Raster (capturer)")) {
    c->sess_pushed.fetch_sub(1, std::memory_order_acq_rel);
    shm_abort_capturer(fps);
    capturer_join_workers(c);
    return;
  }

  const uint32_t pushed = c->sess_pushed.load(std::memory_order_acquire);
  const uint32_t qdep = (uint32_t)c->frame_q->size();
  const int ji = c->jpeg_inflight.load(std::memory_order_acquire);
  const uint32_t inflight = qdep + (uint32_t)std::max(0, ji);
  std::memcpy(shm + kShmCapturerInflightOff, &inflight, sizeof(inflight));
  capturer_publish_queue_depth(c);

  if (pushed >= tot) {
    const uint32_t p_write = kCapturerPhaseWriting;
    std::memcpy(shm + kShmCapturerPhaseOff, &p_write, sizeof(p_write));
  }
}
