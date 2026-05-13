#pragma once
/** FFmpeg RTSP encoder stage (NVENC / libx264). Include after queue typedefs in `plotter_common.cuh`. */

#include <cstddef>
#include <memory>

/** FFmpeg RTSP child (NVENC or libx264): two worker threads read **rgba_q** (optional) and **fifo_images_path**
 *  (optional) and serialize stdin writes. Pass `nullptr` / empty string to disable an input; the corresponding
 *  thread exits immediately. Destruction joins threads and closes the FFmpeg pipe / child process. */
class EncoderStage {
 public:
  EncoderStage(const char* rtsp_url, int frame_w, int frame_h, int encoder_fps, const char* preset, EncodePolicy enc_pol,
               int pipeline_fps, bool use_nvenc, void* fps_base, size_t rgba_npixels, size_t rgba_bytes,
               RgbaFrameQueue* rgba_q, const char* fifo_images_path, bool fapi_slot_path, bool resource_grid_slot_iq_stream);
  ~EncoderStage();
  EncoderStage(const EncoderStage&) = delete;
  EncoderStage& operator=(const EncoderStage&) = delete;
  EncoderStage(EncoderStage&&) = delete;
  EncoderStage& operator=(EncoderStage&&) = delete;

  /** False if the FFmpeg child could not be spawned (no worker threads were started). */
  bool ok() const;
  /** Waits until both worker threads have exited (normally after `g_run` becomes false or FIFO EOF). */
  void join();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
