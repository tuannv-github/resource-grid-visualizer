/*
 * Encoder stage: FFmpeg H.264 child (NVENC or libx264) + dual-input worker threads.
 */
#include "plotter_common.cuh"
#include "plotter_encoder.cuh"
#include "plotter_posix_mq.cuh"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

int fork_ffmpeg_rtsp_stdin(const char* url, int w, int h, int encoder_fps, const char* preset, pid_t* pid_out,
                           EncodePolicy policy, int pipeline_fps, bool use_nvenc) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    perror("pipe");
    return -1;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  if (pid == 0) {
    dup2(pipefd[0], STDIN_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    char wxh[48], in_rs[32], gop[32], vf[64];
    snprintf(wxh, sizeof wxh, "%dx%d", w, h);
    const int gopv = std::max(encoder_fps / 2, 30);
    snprintf(gop, sizeof gop, "%d", gopv);
    char x264p[96];
    snprintf(x264p, sizeof x264p, "keyint=%d:min-keyint=%d:scenecut=0", gopv, gopv);
    if (use_nvenc) {
      if (policy == EncodePolicy::FfmpegFilter) {
        const int in_r = std::min(600, std::max(1, pipeline_fps));
        snprintf(in_rs, sizeof in_rs, "%d", in_r);
        snprintf(vf, sizeof vf, "fps=%d:round=floor", encoder_fps);
        execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", wxh, "-r", in_rs, "-i", "-", "-vf", vf, "-an", "-c:v", "h264_nvenc", "-preset", preset,
               "-pix_fmt", "yuv420p", "-g", gop, "-forced-idr", "1", "-f", "rtsp", "-rtsp_transport", "tcp", url,
               nullptr);
      } else {
        snprintf(in_rs, sizeof in_rs, "%d", encoder_fps);
        execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", wxh, "-r", in_rs, "-i", "-", "-an", "-c:v", "h264_nvenc", "-preset", preset, "-pix_fmt", "yuv420p",
               "-g", gop, "-forced-idr", "1", "-f", "rtsp", "-rtsp_transport", "tcp", url, nullptr);
      }
    } else if (policy == EncodePolicy::FfmpegFilter) {
      const int in_r = std::min(600, std::max(1, pipeline_fps));
      snprintf(in_rs, sizeof in_rs, "%d", in_r);
      snprintf(vf, sizeof vf, "fps=%d:round=floor", encoder_fps);
      execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", wxh, "-r", in_rs, "-i", "-", "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune",
             "zerolatency", "-pix_fmt", "yuv420p", "-bf", "0", "-g", gop, "-keyint_min", gop, "-x264-params", x264p,
             "-f", "rtsp", "-rtsp_transport", "tcp", url, nullptr);
    } else {
      snprintf(in_rs, sizeof in_rs, "%d", encoder_fps);
      execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", wxh, "-r", in_rs, "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
             "-pix_fmt", "yuv420p", "-bf", "0", "-g", gop, "-keyint_min", gop, "-x264-params", x264p, "-f", "rtsp",
             "-rtsp_transport", "tcp", url, nullptr);
    }
    perror("execlp ffmpeg");
    _exit(127);
  }
  close(pipefd[0]);
  *pid_out = pid;
  return pipefd[1];
}

bool write_rgba_to_ffmpeg_fd(int ff_pipe_fd, void* fps_base, size_t rgba_bytes, const uint8_t* rgba, size_t nbytes) {
  if (ff_pipe_fd < 0 || !rgba) return false;
  if (nbytes != rgba_bytes) {
    fprintf(stderr,
            "resgrid_plotter: refuse partial/wrong RGBA to ffmpeg (%zu bytes, expected %zu); stopping encode "
            "(OOM/corruption?)\n",
            nbytes, rgba_bytes);
    return false;
  }
  size_t off = 0;
  while (off < nbytes) {
    const ssize_t nw = ::write(ff_pipe_fd, rgba + off, nbytes - off);
    if (nw < 0) {
      if (errno == EINTR) continue;
      fprintf(stderr, "resgrid_plotter: ffmpeg stdin write error after %zu/%zu bytes: %s\n", off, nbytes,
              strerror(errno));
      return false;
    }
    if (nw == 0) {
      fprintf(stderr, "resgrid_plotter: ffmpeg stdin write stalled at %zu/%zu bytes\n", off, nbytes);
      return false;
    }
    off += (size_t)nw;
  }
  g_stage_output_count_encoder.fetch_add(1, std::memory_order_relaxed);
  fps_shm_store_u64(fps_base, 32, g_stage_output_count_encoder.load(std::memory_order_relaxed));
  return true;
}

}  // namespace

struct EncoderStage::Impl {
  EncoderStageCtx ctx_{};
  pid_t ff_pid_ = -1;
  bool spawn_ok_ = false;
  std::mutex write_mtx_;
  std::string mq_name_;
  std::thread q_th_;
  std::thread m_th_;

  bool write_rgba_locked(const uint8_t* rgba, size_t nbytes) {
    std::lock_guard<std::mutex> lk(write_mtx_);
    return write_rgba_to_ffmpeg_fd(ctx_.ff_pipe_fd, ctx_.fps_base, ctx_.rgba_bytes, rgba, nbytes);
  }

  void queue_thread() {
    if (!ctx_.rgba_q) return;
    fprintf(stderr, "resgrid_plotter: Encoder queue-input thread started\n");
    EncoderStageCtx* const ec = &ctx_;
    std::vector<uint8_t> last_enc_rgba(ec->rgba_bytes);
    bool last_enc_valid = false;
    const int64_t encoder_period_ns = static_cast<int64_t>(
        1e9 / static_cast<double>(std::max(1, ec->encoder_fps)) + 0.5);
    const auto enc_period = std::chrono::nanoseconds(encoder_period_ns);
    auto next_enc = std::chrono::steady_clock::now();
    RgbaFrameQueue* rgba_q = ec->rgba_q;

    if (ec->enc_pol == EncodePolicy::FfmpegFilter) {
      while (g_run.load(std::memory_order_relaxed)) {
        std::vector<uint8_t> popb;
        bool popped = false;
        while (g_run.load(std::memory_order_relaxed) && rgba_q->try_pop(popb)) {
          popped = true;
          last_enc_rgba = popb;
          last_enc_valid = true;
          if (!write_rgba_locked(popb.data(), popb.size())) {
            g_run = false;
            break;
          }
        }
        if (!g_run.load(std::memory_order_relaxed)) break;
        const auto now_ff = std::chrono::steady_clock::now();
        const bool slot_idle = (ec->fapi_slot_path || ec->resource_grid_slot_iq_stream) && !popped;
        if (now_ff >= next_enc) {
          if (slot_idle && last_enc_valid) {
            if (!write_rgba_locked(last_enc_rgba.data(), last_enc_rgba.size())) {
              g_run = false;
              break;
            }
          }
          next_enc += enc_period;
          while (next_enc <= std::chrono::steady_clock::now()) next_enc += enc_period;
        } else if (slot_idle) {
          std::this_thread::sleep_until(next_enc);
        } else if (!popped) {
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      }
    } else {
      while (g_run.load(std::memory_order_relaxed)) {
        std::vector<uint8_t> tmp;
        bool any_pop = false;
        while (rgba_q->try_pop(tmp)) {
          any_pop = true;
          if (ec->enc_pol == EncodePolicy::Mean) {
            for (size_t p = 0; p < ec->rgba_npixels; ++p) {
              const size_t i = p * 4u;
              ec->mean_acc_r[p] += plotter_srgb_u8_to_linear_f(tmp[i]);
              ec->mean_acc_g[p] += plotter_srgb_u8_to_linear_f(tmp[i + 1]);
              ec->mean_acc_b[p] += plotter_srgb_u8_to_linear_f(tmp[i + 2]);
              ec->mean_acc_a[p] += (uint32_t)tmp[i + 3];
            }
            ++ec->rgba_mean_count;
          } else {
            last_enc_rgba = std::move(tmp);
            last_enc_valid = true;
          }
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_enc) {
          if (ec->enc_pol == EncodePolicy::Mean && ec->rgba_mean_count > 0) {
            const uint64_t n = ec->rgba_mean_count;
            const float inv = 1.0f / (float)n;
            for (size_t p = 0; p < ec->rgba_npixels; ++p) {
              const size_t i = p * 4u;
              last_enc_rgba[i] = plotter_linear_f_to_srgb_u8(ec->mean_acc_r[p] * inv);
              last_enc_rgba[i + 1] = plotter_linear_f_to_srgb_u8(ec->mean_acc_g[p] * inv);
              last_enc_rgba[i + 2] = plotter_linear_f_to_srgb_u8(ec->mean_acc_b[p] * inv);
              last_enc_rgba[i + 3] = (uint8_t)((ec->mean_acc_a[p] + n / 2) / n);
            }
            std::fill(ec->mean_acc_r.begin(), ec->mean_acc_r.end(), 0.f);
            std::fill(ec->mean_acc_g.begin(), ec->mean_acc_g.end(), 0.f);
            std::fill(ec->mean_acc_b.begin(), ec->mean_acc_b.end(), 0.f);
            std::fill(ec->mean_acc_a.begin(), ec->mean_acc_a.end(), 0u);
            ec->rgba_mean_count = 0;
            last_enc_valid = true;
          }
          /* A slot stream slower than encoder_fps used to skip the tick. FFmpeg then
           * stalls and the player freezes. Repeat the latest RGBA so stdin stays at
           * encoder_fps (default 24). Live input rate is unchanged, so the UI can warn. */
          if (last_enc_valid) {
            if (!write_rgba_locked(last_enc_rgba.data(), last_enc_rgba.size())) {
              g_run = false;
              break;
            }
          }
          next_enc += enc_period;
          while (next_enc <= std::chrono::steady_clock::now()) next_enc += enc_period;
        } else {
          if ((ec->fapi_slot_path || ec->resource_grid_slot_iq_stream) && !any_pop)
            std::this_thread::sleep_until(next_enc);
          else if (!any_pop)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      }
    }
    fprintf(stderr, "resgrid_plotter: Encoder queue-input thread exiting\n");
  }

  void mq_images_thread() {
    if (mq_name_.empty()) return;
    fprintf(stderr, "resgrid_plotter: --mq-images opening %s (%zu bytes/frame)\n", mq_name_.c_str(), ctx_.rgba_bytes);
    ResgridMqStreamIn mq;
    if (!mq.open_read(mq_name_.c_str())) {
      perror("resgrid_plotter open --mq-images");
      g_run.store(false, std::memory_order_relaxed);
      return;
    }
    std::vector<uint8_t> buf(ctx_.rgba_bytes);
    while (g_run.load(std::memory_order_relaxed)) {
      if (!mq.read_exact(buf.data(), ctx_.rgba_bytes, "--mq-images rgba")) {
        g_run.store(false, std::memory_order_relaxed);
        break;
      }
      if (!write_rgba_locked(buf.data(), ctx_.rgba_bytes)) {
        g_run.store(false, std::memory_order_relaxed);
        break;
      }
    }
    mq.close();
    fprintf(stderr, "resgrid_plotter: --mq-images feed exiting\n");
  }
};

EncoderStage::EncoderStage(const char* rtsp_url, int frame_w, int frame_h, int encoder_fps, const char* preset,
                   EncodePolicy enc_pol, int pipeline_fps, bool use_nvenc, void* fps_base, size_t rgba_npixels,
                   size_t rgba_bytes, RgbaFrameQueue* rgba_q, const char* mq_images_path, bool fapi_slot_path,
                   bool resource_grid_slot_iq_stream)
    : impl_(std::make_unique<Impl>()) {
  impl_->ctx_.rgba_q = rgba_q;
  impl_->ctx_.ff_pipe_fd = -1;
  impl_->ctx_.enc_pol = enc_pol;
  impl_->ctx_.fapi_slot_path = fapi_slot_path;
  impl_->ctx_.resource_grid_slot_iq_stream = resource_grid_slot_iq_stream;
  impl_->ctx_.encoder_fps = encoder_fps;
  impl_->ctx_.fps = pipeline_fps; /* pipeline tick rate for ffmpeg-filter input `-r` */
  impl_->ctx_.fps_base = fps_base;
  impl_->ctx_.rgba_npixels = rgba_npixels;
  impl_->ctx_.rgba_bytes = rgba_bytes;
  if (enc_pol == EncodePolicy::Mean) {
    impl_->ctx_.mean_acc_r.assign(rgba_npixels, 0.f);
    impl_->ctx_.mean_acc_g.assign(rgba_npixels, 0.f);
    impl_->ctx_.mean_acc_b.assign(rgba_npixels, 0.f);
    impl_->ctx_.mean_acc_a.assign(rgba_npixels, 0u);
  }
  if (mq_images_path && mq_images_path[0]) impl_->mq_name_ = mq_images_path;

  const int fd =
      fork_ffmpeg_rtsp_stdin(rtsp_url, frame_w, frame_h, encoder_fps, preset, &impl_->ff_pid_, enc_pol, pipeline_fps,
                             use_nvenc);
  if (fd < 0) {
    impl_->spawn_ok_ = false;
    return;
  }
  impl_->ctx_.ff_pipe_fd = fd;
  impl_->spawn_ok_ = true;

  impl_->q_th_ = std::thread([p = impl_.get()] { p->queue_thread(); });
  impl_->m_th_ = std::thread([p = impl_.get()] { p->mq_images_thread(); });
}

EncoderStage::~EncoderStage() {
  if (!impl_) return;
  if (impl_->q_th_.joinable()) impl_->q_th_.join();
  if (impl_->m_th_.joinable()) impl_->m_th_.join();
  if (impl_->ctx_.ff_pipe_fd >= 0) {
    close(impl_->ctx_.ff_pipe_fd);
    impl_->ctx_.ff_pipe_fd = -1;
  }
  if (impl_->ff_pid_ > 0) {
    kill(impl_->ff_pid_, SIGTERM);
    waitpid(impl_->ff_pid_, nullptr, 0);
    impl_->ff_pid_ = -1;
  }
}

bool EncoderStage::ok() const { return impl_ && impl_->spawn_ok_; }

void EncoderStage::join() {
  if (!impl_) return;
  if (impl_->q_th_.joinable()) impl_->q_th_.join();
  if (impl_->m_th_.joinable()) impl_->m_th_.join();
}
