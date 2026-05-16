#pragma once
/** Post-CLI validated launch: RGBA-only, IQ + CUDA raster + encode, or both (RGBA muxed on FFmpeg stdin). */

#include <string>

#include "plotter_common.cuh"

struct PlotterLaunchConfig {
  const char* prog = "resgrid_plotter";
  /** True only for **--mq-images** without IQ message queues (encoder-only fast path). */
  bool mq_images_only = false;
  std::string rtsp;
  std::string cuda_shm_name;
  std::string mq_images;
  std::string mq_resource_grid;
  std::string mq_fapi;
  std::string capturer_dir;
  std::string preset = "p4";
  int prb = 273;
  int sym = 14;
  /** Inclusive 0-based ``--start-symbol`` (default 0). */
  int sym_start = 0;
  /** Number of OFDM symbols actually rendered (``--stop-symbol - --start-symbol + 1``; defaults to ``sym``). */
  int sym_count = 14;
  int fps = 200;
  int encoder_fps = 24;
  int qdepth = 32;
  int raster_threads = 16;
  float mag_scale = 1.f;
  EncodePolicy enc_pol = EncodePolicy::Last;
  bool video_encoder_nvenc = true;
  bool square_canvas = true;
  bool square_letterbox = false;
  bool have_fapi = false;
  bool have_res = false;
  bool resource_grid_slot_stream = false;
  int frame_w = 0;
  int frame_h = 0;
  int off_x = 0;
  int off_y = 0;
  int content_w = 0;
  int content_h = 0;
  float cs_x = 0.f;
  float cs_y = 0.f;
};

/** RGBA-only, full IQ pipeline, or IQ + optional **--mq-images** (implementation in `plotter_pipeline.cu`). */
int plotter_run_launch(const PlotterLaunchConfig& cfg);
