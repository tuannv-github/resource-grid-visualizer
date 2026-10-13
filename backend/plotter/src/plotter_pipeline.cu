/*
 * Runtime launch: SHM map, stats thread, FIFO-only path, IQ + CUDA raster + assembler + encoder join/shutdown.
 */
#include "plotter_posix_mq.cuh"
#include "plotter_pipeline.cuh"

#include <algorithm>
#include <fcntl.h>
#include <memory>
#include <signal.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

static void on_sig(int) { g_run = false; }

struct FpsShmMap {
  void* addr = nullptr;
  size_t len = 0;
  FpsShmMap() = default;
  explicit FpsShmMap(const std::string& name) {
    if (name.empty()) return;
    int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
      perror("resgrid_plotter shm_open --cuda-shm");
      return;
    }
    len = kIqCudaShmBytes;
    addr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
      perror("resgrid_plotter mmap --cuda-shm");
      addr = nullptr;
      len = 0;
    }
  }
  ~FpsShmMap() {
    if (addr && addr != MAP_FAILED) munmap(addr, len);
  }
  void* ptr() const {
    if (!addr || addr == MAP_FAILED) return nullptr;
    return addr;
  }
};

static void pipeline_stage_output_stats_main(void* fps_base, bool internal_synth_active, CapturerStage* capturer_ptr,
                                             bool mq_images_telemetry) {
  constexpr size_t kAssemblerIn = 40;
  constexpr size_t kAssemblerOut = 48;
  constexpr size_t kRasterIn = 56;
  constexpr size_t kRasterOut = 64;
  constexpr size_t kEncoderIn = 72;
  constexpr size_t kEncoderOut = 80;
  uint64_t p0 = g_producer_frames.load(std::memory_order_relaxed);
  uint64_t ain0 = g_assembler_input_msgs.load(std::memory_order_relaxed);
  uint64_t a_out0 = g_stage_output_count_assembler.load(std::memory_order_relaxed);
  uint64_t r_out0 = g_stage_output_count_raster.load(std::memory_order_relaxed);
  uint64_t e_out0 = g_stage_output_count_encoder.load(std::memory_order_relaxed);
  auto t0 = std::chrono::steady_clock::now();
  while (g_run.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!g_run.load(std::memory_order_relaxed)) break;
    if (capturer_ptr) capturer_ptr->poll_idle_cleanup();
    auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t p1 = g_producer_frames.load(std::memory_order_relaxed);
    const uint64_t ain1 = g_assembler_input_msgs.load(std::memory_order_relaxed);
    const uint64_t a_out1 = g_stage_output_count_assembler.load(std::memory_order_relaxed);
    const uint64_t r_out1 = g_stage_output_count_raster.load(std::memory_order_relaxed);
    const uint64_t e_out1 = g_stage_output_count_encoder.load(std::memory_order_relaxed);
    if (dt > 0.05 && fps_base) {
      if (internal_synth_active) fps_shm_store(fps_base, 0, (double)(p1 - p0) / dt);
      const double assembler_out_hz = (double)(a_out1 - a_out0) / dt;
      const double raster_out_hz = (double)(r_out1 - r_out0) / dt;
      const double enc_out_hz = (double)(e_out1 - e_out0) / dt;
      if (mq_images_telemetry) {
        fps_shm_store(fps_base, 8, enc_out_hz);
        fps_shm_store(fps_base, kAssemblerIn, 0.0);
        fps_shm_store(fps_base, kAssemblerOut, 0.0);
        fps_shm_store(fps_base, kRasterIn, enc_out_hz);
        fps_shm_store(fps_base, kRasterOut, enc_out_hz);
        fps_shm_store(fps_base, kEncoderIn, enc_out_hz);
        fps_shm_store(fps_base, kEncoderOut, enc_out_hz);
      } else {
        fps_shm_store(fps_base, 8, raster_out_hz);
        fps_shm_store(fps_base, kAssemblerIn, (double)(ain1 - ain0) / dt);
        fps_shm_store(fps_base, kAssemblerOut, assembler_out_hz);
        fps_shm_store(fps_base, kRasterIn, assembler_out_hz);
        fps_shm_store(fps_base, kRasterOut, raster_out_hz);
        fps_shm_store(fps_base, kEncoderIn, raster_out_hz);
        fps_shm_store(fps_base, kEncoderOut, enc_out_hz);
      }
      const uint32_t held = g_input_held.load(std::memory_order_relaxed);
      std::memcpy(static_cast<char*>(fps_base) + kShmInputHeldOff, &held, sizeof(held));
    }
    p0 = p1;
    ain0 = ain1;
    a_out0 = a_out1;
    r_out0 = r_out1;
    e_out0 = e_out1;
    t0 = t1;
  }
}

int plotter_run_launch(const PlotterLaunchConfig& cfg) {
  const size_t rgba_npixels = (size_t)cfg.frame_w * (size_t)cfg.frame_h;
  const size_t rgba_bytes = rgba_npixels * 4u;

  if (cfg.mq_images_only) {
    if (cfg.enc_pol == EncodePolicy::Mean) {
      fprintf(stderr, "%s: --mq-images is incompatible with --encode-policy mean\n", cfg.prog);
      return 1;
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    FpsShmMap fps_map_img(cfg.cuda_shm_name);
    void* fps_base_img = fps_map_img.ptr();
    fprintf(stderr,
            "resgrid_plotter: --mq-images %dx%d encoder_fps=%d %s -> %s\n", cfg.frame_w, cfg.frame_h,
            cfg.encoder_fps, cfg.video_encoder_nvenc ? "h264_nvenc" : "libx264", cfg.rtsp.c_str());
    EncoderStage encoder_stage(cfg.rtsp.c_str(), cfg.frame_w, cfg.frame_h, cfg.encoder_fps, cfg.preset.c_str(), cfg.enc_pol,
                    cfg.fps, cfg.video_encoder_nvenc, fps_base_img, rgba_npixels, rgba_bytes, nullptr,
                    cfg.mq_images.c_str(), false, false);
    if (!encoder_stage.ok()) {
      fprintf(stderr, "failed to spawn ffmpeg\n");
      return 1;
    }
    std::thread stats_img(pipeline_stage_output_stats_main, fps_base_img, false, nullptr, true);
    encoder_stage.join();
    g_run = false;
    if (stats_img.joinable()) stats_img.join();
    return 0;
  }

  const int ncells = cfg.prb * cfg.sym;
  const int rt = std::max(1, std::min(64, cfg.raster_threads));

  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  ResgridMqStreamIn* iq_mq_full_grid = nullptr;
  ResgridMqStreamIn* iq_mq_sidecar = nullptr;
  std::unique_ptr<InternalFrameQueue> internal_q;
  std::unique_ptr<FullGridFifoQueue> full_grid_fifo_q;
  std::unique_ptr<AssembledSlotQueue> slot_assembled_q;
  std::unique_ptr<ChannelModeIqSidecarQueue> iq_sidecar_q;

  std::vector<float2*> d_z(rt, nullptr);
  std::vector<float*> d_mag(rt, nullptr);
  std::vector<uint8_t*> d_chan(rt, nullptr);
  std::vector<uint32_t*> d_cell_rnti(rt, nullptr);
  std::vector<uint8_t*> d_rgba(rt, nullptr);
  auto free_raster_cuda_bufs = [&]() {
    for (int i = 0; i < rt; ++i) {
      if (d_z[i]) cudaFree(d_z[i]);
      if (d_mag[i]) cudaFree(d_mag[i]);
      if (d_chan[i]) cudaFree(d_chan[i]);
      if (d_cell_rnti[i]) cudaFree(d_cell_rnti[i]);
      if (d_rgba[i]) cudaFree(d_rgba[i]);
      d_z[i] = nullptr;
      d_mag[i] = nullptr;
      d_chan[i] = nullptr;
      d_cell_rnti[i] = nullptr;
      d_rgba[i] = nullptr;
    }
  };
  for (int i = 0; i < rt; ++i) {
    cuda_check(cudaMalloc(&d_z[i], ncells * sizeof(float2)), "cudaMalloc z");
    cuda_check(cudaMalloc(&d_mag[i], ncells * sizeof(float)), "cudaMalloc mag");
    cuda_check(cudaMalloc(&d_chan[i], (size_t)ncells * sizeof(uint8_t)), "cudaMalloc chan");
    cuda_check(cudaMalloc(&d_cell_rnti[i], (size_t)ncells * sizeof(uint32_t)), "cudaMalloc cell_rnti");
    cuda_check(cudaMalloc(&d_rgba[i], (size_t)cfg.frame_w * cfg.frame_h * 4), "cudaMalloc rgba");
  }

  FpsShmMap fps_map(cfg.cuda_shm_name);
  void* fps_base = fps_map.ptr();
  g_plotter_fps_shm = fps_base;
  struct ClearPlotterFpsShm {
    ~ClearPlotterFpsShm() { g_plotter_fps_shm = nullptr; }
  } clear_plotter_fps_shm;

  if (cfg.have_fapi) {
    slot_assembled_q = std::make_unique<AssembledSlotQueue>(std::max(1, cfg.qdepth));
    if (cfg.have_res) {
      iq_sidecar_q = std::make_unique<ChannelModeIqSidecarQueue>(cfg.qdepth);
      iq_mq_sidecar = AssemblerStage::assembler_open_iq_mq_reader(
          cfg.mq_resource_grid.c_str(), "resource-grid MQ as FAPI I/Q sidecar (one grid per message)");
      if (!iq_mq_sidecar) {
        free_raster_cuda_bufs();
        return 1;
      }
    }
  } else if (cfg.have_res) {
    full_grid_fifo_q = std::make_unique<FullGridFifoQueue>(cfg.qdepth);
    iq_mq_full_grid = AssemblerStage::assembler_open_iq_mq_reader(cfg.mq_resource_grid.c_str(),
                                                                  "full-grid resource-grid MQ (one grid per message)");
    if (!iq_mq_full_grid) {
      free_raster_cuda_bufs();
      return 1;
    }
  } else {
    internal_q = std::make_unique<InternalFrameQueue>(cfg.qdepth);
  }

  cudaStream_t raster_cuda_stream = nullptr;
  std::vector<cudaStream_t> raster_streams;
  if (rt == 1) {
    cuda_check(cudaStreamCreateWithFlags(&raster_cuda_stream, cudaStreamNonBlocking), "cudaStreamCreate raster");
  } else {
    raster_streams.resize((size_t)rt);
    for (int i = 0; i < rt; ++i)
      cuda_check(cudaStreamCreateWithFlags(&raster_streams[(size_t)i], cudaStreamNonBlocking),
                 "cudaStreamCreate raster pool");
  }

  const char* pol_name = "last";
  if (cfg.enc_pol == EncodePolicy::Mean) pol_name = "mean";
  else if (cfg.enc_pol == EncodePolicy::FfmpegFilter) pol_name = "ffmpeg";
  fprintf(stderr,
          "resgrid_plotter PRB=%d symbols=%d sym_window=[%d,%d] (%d cols) content=%dx%d cs=(%.4f,%.4f) square=%d "
          "letterbox=%d mq_resource_grid=%d "
          "mq_fapi=%d internal_synth=%d fapi_slot_path=%d rg_slot_stream=%d raster_threads=%d -> %dx%d pipeline_fps=%d "
          "encoder_fps=%d "
          "encode_policy=%s "
          "video_codec=%s mag_scale=%g -> %s\n",
          cfg.prb, cfg.sym, cfg.sym_start, cfg.sym_start + cfg.sym_count - 1, cfg.sym_count, cfg.content_w,
          cfg.content_h, (double)cfg.cs_x, (double)cfg.cs_y, cfg.square_canvas ? 1 : 0,
          (cfg.square_canvas && cfg.square_letterbox) ? 1 : 0, cfg.have_res ? 1 : 0, cfg.have_fapi ? 1 : 0,
          internal_q ? 1 : 0, slot_assembled_q ? 1 : 0, cfg.resource_grid_slot_stream ? 1 : 0, rt, cfg.frame_w,
          cfg.frame_h, cfg.fps, cfg.encoder_fps, pol_name, cfg.video_encoder_nvenc ? "nvenc" : "libx264",
          (double)cfg.mag_scale, cfg.rtsp.c_str());

  const dim3 block_mag(256, 1, 1);
  const dim3 grid_mag((ncells + block_mag.x - 1) / block_mag.x, 1, 1);
  const dim3 block2(16, 16, 1);
  const dim3 grid2((cfg.frame_w + block2.x - 1) / block2.x, (cfg.frame_h + block2.y - 1) / block2.y, 1);

  std::unique_ptr<RgbaFrameQueue> rgba_q = std::make_unique<RgbaFrameQueue>(cfg.qdepth);
  std::unique_ptr<CapturerFrameQueue> capturer_q;
  if (!cfg.capturer_dir.empty()) {
    unsigned hc = std::thread::hardware_concurrency();
    const int capturer_cap = std::max(cfg.qdepth, (int)((hc == 0 ? 4u : hc) * 8));
    capturer_q = std::make_unique<CapturerFrameQueue>(capturer_cap);
  }

  CapturerStage capturer_stage{};
  if (!cfg.capturer_dir.empty()) {
    capturer_stage.capturer_configure(&cfg.capturer_dir, fps_base, cfg.frame_w, cfg.frame_h, capturer_q.get());
  }
  CapturerStage* const capturer_ptr = cfg.capturer_dir.empty() ? nullptr : &capturer_stage;

  RasterStage raster_stage;
  raster_stage.raster_configure(rt, rgba_bytes, internal_q.get(), full_grid_fifo_q.get(), slot_assembled_q.get(),
                                iq_sidecar_q.get(), cfg.resource_grid_slot_stream, rgba_q.get(), fps_base, ncells,
                                cfg.prb, cfg.sym, cfg.sym_start, cfg.sym_count, cfg.frame_w, cfg.frame_h, cfg.off_x,
                                cfg.off_y, cfg.content_w, cfg.content_h, cfg.mag_scale, grid_mag, block_mag, grid2,
                                block2, capturer_ptr);
  for (int i = 0; i < rt; ++i)
    raster_stage.raster_set_thread_cuda(i, d_z[(size_t)i], d_mag[(size_t)i], d_chan[(size_t)i], d_cell_rnti[(size_t)i],
                                 d_rgba[(size_t)i], (rt == 1) ? raster_cuda_stream : raster_streams[(size_t)i]);

  std::unique_ptr<RasterWorkQueue> raster_work_q;
  if (rt > 1) {
    raster_work_q = std::make_unique<RasterWorkQueue>(std::max(cfg.qdepth, rt * 2));
    raster_stage.raster_configure_pool(raster_work_q.get());
  }

  if (full_grid_fifo_q && !cfg.have_fapi && iq_mq_full_grid)
    raster_stage.raster_attach_full_grid_mq_reader(iq_mq_full_grid, cfg.resource_grid_slot_stream);

  AssemblerStage assembler_stage;
  std::string mq_fapi_arg = cfg.mq_fapi;
  assembler_stage.assembler_configure(internal_q.get(), fps_base, cfg.fps, ncells, cfg.prb, cfg.sym,
                             slot_assembled_q.get(), &mq_fapi_arg, iq_sidecar_q.get(), iq_mq_sidecar);

  std::thread stats_th(pipeline_stage_output_stats_main, fps_base, internal_q.get() != nullptr, capturer_ptr, false);

  const char* const mq_img_arg = cfg.mq_images.empty() ? "" : cfg.mq_images.c_str();
  std::unique_ptr<EncoderStage> encoder_stage = std::make_unique<EncoderStage>(
      cfg.rtsp.c_str(), cfg.frame_w, cfg.frame_h, cfg.encoder_fps, cfg.preset.c_str(), cfg.enc_pol, cfg.fps,
      cfg.video_encoder_nvenc, fps_base, rgba_npixels, rgba_bytes, rgba_q.get(), mq_img_arg,
      slot_assembled_q.get() != nullptr, cfg.resource_grid_slot_stream);
  if (!encoder_stage->ok()) {
    fprintf(stderr, "failed to spawn ffmpeg\n");
    delete iq_mq_full_grid;
    iq_mq_full_grid = nullptr;
    delete iq_mq_sidecar;
    iq_mq_sidecar = nullptr;
    free_raster_cuda_bufs();
    return 1;
  }

  std::thread assembler_stage_th([&assembler_stage]() { assembler_stage.assembler_run(); });
  std::thread raster_stage_th([&raster_stage]() { raster_stage.raster_run(); });

  encoder_stage->join();
  raster_stage_th.join();
  capturer_stage.join();
  assembler_stage_th.join();

  if (stats_th.joinable()) stats_th.join();
  delete iq_mq_full_grid;
  delete iq_mq_sidecar;

  encoder_stage.reset();

  if (rt == 1) {
    if (raster_cuda_stream) {
      cudaStreamSynchronize(raster_cuda_stream);
      cudaStreamDestroy(raster_cuda_stream);
    }
  } else {
    for (int i = 0; i < rt; ++i) {
      cudaStreamSynchronize(raster_streams[(size_t)i]);
      cudaStreamDestroy(raster_streams[(size_t)i]);
    }
  }

  free_raster_cuda_bufs();
  return 0;
}
