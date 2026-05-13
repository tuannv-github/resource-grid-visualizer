/*
 * Encode stage: FFmpeg NVENC child + stdin writes.
 */
#include "plotter_common.cuh"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <unistd.h>

int spawn_ffmpeg_nvenc_rtsp(const char* url, int w, int h, int encoder_fps, const char* preset, pid_t* pid_out,
                            EncodePolicy policy, int pipeline_fps) {
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
    if (policy == EncodePolicy::FfmpegFilter) {
      const int in_r = std::min(600, std::max(1, pipeline_fps));
      snprintf(in_rs, sizeof in_rs, "%d", in_r);
      snprintf(vf, sizeof vf, "fps=%d:round=floor", encoder_fps);
      execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", wxh, "-r", in_rs, "-i", "-", "-vf", vf, "-an", "-c:v", "h264_nvenc", "-preset", preset, "-pix_fmt",
             "yuv420p", "-g", gop, "-forced-idr", "1", "-f", "rtsp", "-rtsp_transport", "tcp", url, nullptr);
    } else {
      snprintf(in_rs, sizeof in_rs, "%d", encoder_fps);
      execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", wxh, "-r", in_rs, "-i", "-", "-an", "-c:v", "h264_nvenc", "-preset", preset, "-pix_fmt", "yuv420p",
             "-g", gop, "-forced-idr", "1", "-f", "rtsp", "-rtsp_transport", "tcp", url, nullptr);
    }
    perror("execlp ffmpeg");
    _exit(127);
  }
  close(pipefd[0]);
  *pid_out = pid;
  return pipefd[1];
}

bool write_rgba_to_ffmpeg(EncodeStageCtx* ec, const uint8_t* rgba, size_t nbytes) {
  const int fd = ec->ff_pipe_fd;
  void* const fps_base = ec->fps_base;
  if (fd < 0 || !rgba) return false;
  if (nbytes != ec->rgba_bytes) {
    fprintf(stderr,
            "resgrid_plotter: refuse partial/wrong RGBA to ffmpeg (%zu bytes, expected %zu); stopping encode "
            "(OOM/corruption?)\n",
            nbytes, ec->rgba_bytes);
    return false;
  }
  /* Raw `write(2)` avoids stdio lock + fflush per frame (large RGBA goes to NVENC faster under merger load). */
  size_t off = 0;
  while (off < nbytes) {
    const ssize_t nw = ::write(fd, rgba + off, nbytes - off);
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

void encode_stage_main(EncodeStageCtx* ec) {
  fprintf(stderr, "resgrid_plotter: Encode stage thread started\n");
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
        if (!write_rgba_to_ffmpeg(ec, popb.data(), popb.size())) {
          g_run = false;
          break;
        }
        last_enc_rgba = std::move(popb);
        last_enc_valid = true;
      }
      if (!g_run.load(std::memory_order_relaxed)) break;
      const auto now_ff = std::chrono::steady_clock::now();
      const bool idle_hold = ec->dm == 3 && last_enc_valid && !popped;
      if (now_ff >= next_enc) {
        if (idle_hold) {
          if (!write_rgba_to_ffmpeg(ec, last_enc_rgba.data(), last_enc_rgba.size())) {
            g_run = false;
            break;
          }
        }
        next_enc += enc_period;
        while (next_enc <= std::chrono::steady_clock::now()) next_enc += enc_period;
      } else if (idle_hold) {
        std::this_thread::sleep_until(next_enc);
      } else if (!popped) {
        /* Short sleep: merger can fill RgbaFrameQueue between encoder ticks; 1 ms adds avoidable backlog. */
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
        if (last_enc_valid) {
          if (!write_rgba_to_ffmpeg(ec, last_enc_rgba.data(), last_enc_rgba.size())) {
            g_run = false;
            break;
          }
        }
        next_enc += enc_period;
        while (next_enc <= std::chrono::steady_clock::now()) next_enc += enc_period;
      } else {
        if (ec->dm == 3 && last_enc_valid && !any_pop)
          std::this_thread::sleep_until(next_enc);
        else if (!any_pop)
          std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
  }
  fprintf(stderr, "resgrid_plotter: Encode stage thread exiting\n");
}
