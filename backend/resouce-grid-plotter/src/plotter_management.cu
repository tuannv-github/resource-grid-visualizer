/*
 * Management: process globals, SHM map, CLI, stats thread, main encode loop, shutdown.
 */
#include "plotter_common.cuh"

#include <algorithm>
#include <fcntl.h>
#include <mutex>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

std::atomic<bool> g_run{true};
std::atomic<uint64_t> g_producer_frames{0};
std::atomic<uint64_t> g_stage_output_count_assembly{0};
std::atomic<uint64_t> g_stage_output_count_raster{0};
std::atomic<uint64_t> g_stage_output_count_encoder{0};
std::atomic<uint64_t> g_slot_completed_drops{0};
std::atomic<uint64_t> g_grid_frame_queue_drops{0};
std::atomic<uint64_t> g_rgba_frame_queue_drops{0};
std::atomic<uint64_t> g_raster_work_queue_drops{0};
std::atomic<uint64_t> g_capturer_frame_queue_drops{0};
std::atomic<uint64_t> g_assembly_input_msgs{0};

static void on_sig(int) { g_run = false; }

void plotter_warn_queue_drop_oldest(PlotterQueueDropKind kind, const char* queue_name,
                                    const char* producer_stage, uint32_t drops_this_push) {
  if (!drops_this_push || !queue_name) return;
  const size_t i = (size_t)kind;
  if (i >= (size_t)PlotterQueueDropKind::Count_) return;

  /* Mutex + wall interval: lock-free CAS allowed many threads to print in the same instant (e.g. raster merger). */
  static std::mutex mu;
  static std::chrono::steady_clock::time_point last_emit[(size_t)PlotterQueueDropKind::Count_]{};
  constexpr auto kMinInterval = std::chrono::milliseconds(2500);

  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(mu);
  if (now - last_emit[i] < kMinInterval) return;
  last_emit[i] = now;

  const char* who = (producer_stage && producer_stage[0]) ? producer_stage : "unknown";
  if (drops_this_push == 1u)
    fprintf(stderr,
            "resgrid_plotter: warning: producer stage \"%s\" → %s — queue full, dropped oldest output to accept new "
            "input\n",
            who, queue_name);
  else
    fprintf(stderr,
            "resgrid_plotter: warning: producer stage \"%s\" → %s — queue full, dropped %u oldest item(s) to accept "
            "new input\n",
            who, queue_name, drops_this_push);
}

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
    /* Legacy 0…39 + pipeline stage rates (doubles @40…80) + file capturer tail @88… (`kIqCudaShmBytes`). */
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

/**
 * Dedicated statistics thread (~1 s cadence): reads monotonic per-stage **output** counters
 * (`g_stage_output_count_*`, each incremented once per completed unit), computes **Δcount/Δt** as Hz,
 * and writes pipeline doubles into SHM (offsets 8, 40…80).
 */
static void pipeline_stage_output_stats_main(void* fps_base, int data_mode, CapturerCtx* capturer_ptr) {
  constexpr size_t kAsmIn = 40;
  constexpr size_t kAsmOut = 48;
  constexpr size_t kRastIn = 56;
  constexpr size_t kRastOut = 64;
  constexpr size_t kEncIn = 72;
  constexpr size_t kEncOut = 80;
  uint64_t p0 = g_producer_frames.load(std::memory_order_relaxed);
  uint64_t ain0 = g_assembly_input_msgs.load(std::memory_order_relaxed);
  uint64_t a_out0 = g_stage_output_count_assembly.load(std::memory_order_relaxed);
  uint64_t r_out0 = g_stage_output_count_raster.load(std::memory_order_relaxed);
  uint64_t e_out0 = g_stage_output_count_encoder.load(std::memory_order_relaxed);
  auto t0 = std::chrono::steady_clock::now();
  while (g_run.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!g_run.load(std::memory_order_relaxed)) break;
    if (capturer_ptr) capturer_poll_idle_cleanup(capturer_ptr);
    auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t p1 = g_producer_frames.load(std::memory_order_relaxed);
    const uint64_t ain1 = g_assembly_input_msgs.load(std::memory_order_relaxed);
    const uint64_t a_out1 = g_stage_output_count_assembly.load(std::memory_order_relaxed);
    const uint64_t r_out1 = g_stage_output_count_raster.load(std::memory_order_relaxed);
    const uint64_t e_out1 = g_stage_output_count_encoder.load(std::memory_order_relaxed);
    if (dt > 0.05 && fps_base) {
      if (data_mode == 1) fps_shm_store(fps_base, 0, (double)(p1 - p0) / dt);
      const double asm_out_hz = (double)(a_out1 - a_out0) / dt;
      const double raster_out_hz = (double)(r_out1 - r_out0) / dt;
      const double enc_out_hz = (double)(e_out1 - e_out0) / dt;
      fps_shm_store(fps_base, 8, raster_out_hz);
      fps_shm_store(fps_base, kAsmIn, (double)(ain1 - ain0) / dt);
      fps_shm_store(fps_base, kAsmOut, asm_out_hz);
      /* Chained SHM: "input" to each stage = previous stage's output rate (Hz). */
      fps_shm_store(fps_base, kRastIn, asm_out_hz);
      fps_shm_store(fps_base, kRastOut, raster_out_hz);
      fps_shm_store(fps_base, kEncIn, raster_out_hz);
      fps_shm_store(fps_base, kEncOut, enc_out_hz);
    }
    p0 = p1;
    ain0 = ain1;
    a_out0 = a_out1;
    r_out0 = r_out1;
    e_out0 = e_out1;
    t0 = t1;
  }
}

static void usage(const char* av0) {
  fprintf(stderr,
          "Usage: %s [--rtsp URL] [--prb N] [--symbols M] [--cell-pixels P] [--fps F] [--encoder-fps E] [--mag-scale S] "
          "[--preset p4] [--queue-depth D] [--raster-threads N] [--encode-policy POL] [--no-square] [--square-letterbox] "
          "[--data-mode 1|2|3] [--iq-fifo PATH] [--slot-fifo PATH] [--cuda-shm NAME] [--capturer-dir DIR]\n"
          "  --fps F            Pipeline / internal synthetic producer default (Hz); default 200. Also sets --encoder-fps\n"
          "                     if you omit --encoder-fps (backward compatible single-knob).\n"
          "  --encoder-fps E    Target NVENC output rate (Hz); default 24. With --encode-policy last|mean, rawvideo -r\n"
          "                     matches E and the plotter paces stdin. With ffmpeg, rawvideo -r matches --fps (capped 600)\n"
          "                     and FFmpeg -vf fps=E reduces before NVENC.\n"
          "  --encode-policy POL  last (default): each encoder tick sends latest RGBA (subsample in time).\n"
          "                     mean: each tick sends per-pixel color = mean in linear RGB (sRGB EOTF), then OETF;\n"
          "                     alpha = rounded mean of input alphas.\n"
          "                     ffmpeg: write every rendered frame to stdin; FFmpeg fps filter → encoder-fps (idle gaps\n"
          "                     still duplicate at encoder-fps for RTSP keepalive).\n"
          "  --data-mode 1  Internal synthetic grid only (no --iq-fifo / --slot-fifo).\n"
          "  --data-mode 2  Full-grid FIFO: each --iq-fifo message = one complete resource grid → CUDA → NVENC.\n"
          "  --data-mode 3  Slot messages (channel-based): --slot-fifo = per-slot allocations (partial grids);\n"
          "                 assembler builds full grids; optional --iq-fifo = full-grid I/Q sidecar (assembled preferred).\n"
          "  Default mode: 3 if --slot-fifo only; 2 if --iq-fifo; 1 if neither; if both FIFOs set → 3.\n"
          "  --raster-threads N  CUDA raster workers (default 16). N>1: one acquirer preserves slot/sidecar order, N\n"
          "                     GPU workers + merger reorder RGBA for the encoder (cap 64).\n"
          "  --queue-depth D    In-process GridFrame/RgbaFrame queue capacity (default 32); backend env CUDA_IQ_QUEUE_DEPTH.\n",
          av0);
}

int main(int argc, char** argv) {
  std::string rtsp = "rtsp://127.0.0.1:8554/live";
  int prb = 273;
  int sym = 14;
  int cell_px = 6;
  int fps = 200;
  int encoder_fps = 24;
  bool have_fps_cli = false;
  bool have_encoder_fps_cli = false;
  float mag_scale = 1.0f;
  const char* preset = "p4";
  int qdepth = 32;
  int raster_threads = 16;
  bool square_canvas = true;
  bool square_letterbox = false;
  std::string iq_fifo;
  std::string slot_fifo;
  std::string capturer_dir;
  std::string cuda_shm_name;
  std::string encode_policy_arg = "last";
  int data_mode_cli = 0;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rtsp") && i + 1 < argc)
      rtsp = argv[++i];
    else if (!strcmp(argv[i], "--prb") && i + 1 < argc)
      prb = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--symbols") && i + 1 < argc)
      sym = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cell-pixels") && i + 1 < argc)
      cell_px = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--fps") && i + 1 < argc) {
      fps = atoi(argv[++i]);
      have_fps_cli = true;
    } else if (!strcmp(argv[i], "--encoder-fps") && i + 1 < argc) {
      encoder_fps = atoi(argv[++i]);
      have_encoder_fps_cli = true;
    } else if (!strcmp(argv[i], "--mag-scale") && i + 1 < argc)
      mag_scale = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--preset") && i + 1 < argc)
      preset = argv[++i];
    else if (!strcmp(argv[i], "--queue-depth") && i + 1 < argc)
      qdepth = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--raster-threads") && i + 1 < argc)
      raster_threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-square"))
      square_canvas = false;
    else if (!strcmp(argv[i], "--square-letterbox"))
      square_letterbox = true;
    else if (!strcmp(argv[i], "--iq-fifo") && i + 1 < argc)
      iq_fifo = argv[++i];
    else if (!strcmp(argv[i], "--slot-fifo") && i + 1 < argc)
      slot_fifo = argv[++i];
    else if (!strcmp(argv[i], "--data-mode") && i + 1 < argc) {
      data_mode_cli = atoi(argv[++i]);
      if (data_mode_cli < 1 || data_mode_cli > 3) data_mode_cli = 0;
    } else if (!strcmp(argv[i], "--cuda-shm") && i + 1 < argc)
      cuda_shm_name = argv[++i];
    else if (!strcmp(argv[i], "--capturer-dir") && i + 1 < argc)
      capturer_dir = argv[++i];
    else if (!strcmp(argv[i], "--encoder-snap-dir") && i + 1 < argc)
      capturer_dir = argv[++i];
    else if (!strcmp(argv[i], "--encode-policy") && i + 1 < argc)
      encode_policy_arg = argv[++i];
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage(argv[0]);
      return 0;
    }
  }

  if (!have_encoder_fps_cli && have_fps_cli) encoder_fps = fps;

  /* Cap NVENC / encode-stage pacing (stdin wall-clock ticks) for load testing; remove or raise when done. */
  constexpr int kEncoderFpsHardMax = 50;
  if (encoder_fps > kEncoderFpsHardMax) {
    fprintf(stderr, "%s: capping --encoder-fps %d -> %d\n", argv[0], encoder_fps, kEncoderFpsHardMax);
    encoder_fps = kEncoderFpsHardMax;
  }

  const EncodePolicy enc_pol = parse_encode_policy(argv[0], encode_policy_arg.c_str());
  if (enc_pol == EncodePolicy::FfmpegFilter && encoder_fps >= fps)
    fprintf(stderr,
            "%s: warning: --encode-policy ffmpeg with --encoder-fps >= --fps; filter may duplicate/drop oddly "
            "(pipeline_fps=%d encoder_fps=%d)\n",
            argv[0], fps, encoder_fps);

  if (prb < 1 || sym < 1 || cell_px < 1 || fps < 1 || encoder_fps < 1 || mag_scale <= 0.f) {
    usage(argv[0]);
    return 1;
  }

  int dm;
  if (data_mode_cli >= 1 && data_mode_cli <= 3)
    dm = data_mode_cli;
  else if (!iq_fifo.empty() && !slot_fifo.empty())
    dm = 3;
  else if (!iq_fifo.empty())
    dm = 2;
  else if (!slot_fifo.empty())
    dm = 3;
  else
    dm = 1;

  if (dm == 1 && (!iq_fifo.empty() || !slot_fifo.empty())) {
    fprintf(stderr,
            "%s: --data-mode 1 (or default internal-only) requires no --iq-fifo and no --slot-fifo\n", argv[0]);
    return 1;
  }
  if (dm == 2 && iq_fifo.empty()) {
    fprintf(stderr, "%s: --data-mode 2 (full-grid FIFO) requires --iq-fifo\n", argv[0]);
    return 1;
  }
  if (dm == 2 && !slot_fifo.empty()) {
    fprintf(stderr,
            "%s: --data-mode 2 (full-grid FIFO) must not use --slot-fifo (use --data-mode 3 for slot messages)\n",
            argv[0]);
    return 1;
  }
  if (dm == 3 && slot_fifo.empty()) {
    fprintf(stderr, "%s: --data-mode 3 (slot messages / channel-based) requires --slot-fifo\n", argv[0]);
    return 1;
  }

  int frame_w;
  int frame_h;
  int off_x = 0;
  int off_y = 0;
  int content_w;
  int content_h;
  float cs_x;
  float cs_y;

  if (square_canvas) {
    int side = std::max(prb * cell_px, sym * cell_px);
    if (side % 2) ++side;
    frame_w = frame_h = side;
    if (square_letterbox) {
      const float cell_scale = (float)side / (float)std::max(prb, sym);
      content_w = (int)lrintf((float)sym * cell_scale);
      content_h = (int)lrintf((float)prb * cell_scale);
      off_x = (frame_w - content_w) / 2;
      off_y = (frame_h - content_h) / 2;
      cs_x = cs_y = cell_scale;
    } else {
      off_x = off_y = 0;
      content_w = content_h = side;
      cs_x = (float)side / (float)sym;
      cs_y = (float)side / (float)prb;
    }
  } else {
    cs_x = cs_y = (float)cell_px;
    content_w = sym * cell_px;
    content_h = prb * cell_px;
    frame_w = content_w + (content_w & 1);
    frame_h = content_h + (content_h & 1);
    off_x = off_y = 0;
  }

  /* h264_nvenc: each side >= ~128 on many GPUs; width/height should be multiples of 16 for some drivers
   * (e.g. 1638-high frames fail InitializeEncoder even when width is 128). Pad/center then round up to 16. */
  constexpr int kNvencMinDim = 128;
  auto align_up_16 = [](int x) { return (x + 15) & ~15; };
  {
    int need_w = std::max(0, kNvencMinDim - frame_w);
    int need_h = std::max(0, kNvencMinDim - frame_h);
    if (need_w || need_h) {
      off_x += need_w / 2;
      off_y += need_h / 2;
      frame_w += need_w;
      frame_h += need_h;
    }
    const int aw = align_up_16(frame_w);
    const int ah = align_up_16(frame_h);
    const int exw = aw - frame_w;
    const int exh = ah - frame_h;
    if (exw || exh) {
      off_x += exw / 2;
      off_y += exh / 2;
      frame_w = aw;
      frame_h = ah;
    }
  }

  const int ncells = prb * sym;
  const int rt = std::max(1, std::min(64, raster_threads));

  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  int iq_fd_full_grid = -1;
  int iq_fd_sidecar = -1;
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
    cuda_check(cudaMalloc(&d_rgba[i], (size_t)frame_w * frame_h * 4), "cudaMalloc rgba");
  }

  FpsShmMap fps_map(cuda_shm_name);
  void* fps_base = fps_map.ptr();

  if (dm == 1) {
    internal_q = std::make_unique<InternalFrameQueue>(qdepth);
  } else if (dm == 2) {
    full_grid_fifo_q = std::make_unique<FullGridFifoQueue>(qdepth);
    iq_fd_full_grid = open_iq_fifo_read_end(iq_fifo.c_str(), "full-grid IQ FIFO (one grid per message)");
    if (iq_fd_full_grid < 0) {
      free_raster_cuda_bufs();
      return 1;
    }
  } else {
    slot_assembled_q = std::make_unique<AssembledSlotQueue>(std::max(1, qdepth));
    if (!iq_fifo.empty()) {
      iq_sidecar_q = std::make_unique<ChannelModeIqSidecarQueue>(qdepth);
      iq_fd_sidecar = open_iq_fifo_read_end(iq_fifo.c_str(), "channel-based IQ sidecar (full grid per message)");
      if (iq_fd_sidecar < 0) {
        free_raster_cuda_bufs();
        return 1;
      }
    }
  }

  const size_t rgba_npixels = (size_t)frame_w * (size_t)frame_h;
  const size_t rgba_bytes = rgba_npixels * 4u;

  pid_t ff_pid = -1;
  const int ff_pipe = spawn_ffmpeg_nvenc_rtsp(rtsp.c_str(), frame_w, frame_h, encoder_fps, preset, &ff_pid, enc_pol,
                                              fps);
  if (ff_pipe < 0) {
    fprintf(stderr, "failed to spawn ffmpeg\n");
    if (iq_fd_full_grid >= 0) close(iq_fd_full_grid);
    if (iq_fd_sidecar >= 0) close(iq_fd_sidecar);
    free_raster_cuda_bufs();
    return 1;
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
  if (enc_pol == EncodePolicy::Mean) pol_name = "mean";
  else if (enc_pol == EncodePolicy::FfmpegFilter) pol_name = "ffmpeg";
  fprintf(stderr,
          "resgrid_plotter PRB=%d symbols=%d content=%dx%d cs=(%.4f,%.4f) square=%d letterbox=%d iq_fifo=%d "
          "slot_fifo=%d data_mode=%d raster_threads=%d -> %dx%d pipeline_fps=%d encoder_fps=%d encode_policy=%s "
          "mag_scale=%g -> %s\n",
          prb, sym, content_w, content_h, (double)cs_x, (double)cs_y, square_canvas ? 1 : 0,
          (square_canvas && square_letterbox) ? 1 : 0, iq_fifo.empty() ? 0 : 1, slot_fifo.empty() ? 0 : 1, dm, rt,
          frame_w, frame_h, fps, encoder_fps, pol_name, (double)mag_scale, rtsp.c_str());

  const dim3 block_mag(256, 1, 1);
  const dim3 grid_mag((ncells + block_mag.x - 1) / block_mag.x, 1, 1);
  const dim3 block2(16, 16, 1);
  const dim3 grid2((frame_w + block2.x - 1) / block2.x, (frame_h + block2.y - 1) / block2.y, 1);

  std::unique_ptr<RgbaFrameQueue> rgba_q = std::make_unique<RgbaFrameQueue>(qdepth);
  std::unique_ptr<CapturerFrameQueue> capturer_q;
  if (!capturer_dir.empty()) {
    unsigned hc = std::thread::hardware_concurrency();
    const int capturer_cap = std::max(qdepth, (int)((hc == 0 ? 4u : hc) * 8));
    capturer_q = std::make_unique<CapturerFrameQueue>(capturer_cap);
  }

  std::vector<RasterThreadCtx> raster_ctxs((size_t)rt);
  std::unique_ptr<RasterWorkQueue> raster_work_q;
  RasterPoolCtrl raster_pool{};
  RasterAcquireParams raster_ap{};

  for (int i = 0; i < rt; ++i) {
    RasterThreadCtx& rc = raster_ctxs[(size_t)i];
    rc.dm = dm;
    rc.internal_q = internal_q.get();
    rc.full_grid_q = full_grid_fifo_q.get();
    rc.assembled_q = slot_assembled_q.get();
    rc.sidecar_q = iq_sidecar_q.get();
    rc.rgba_out = rgba_q.get();
    rc.fps_base = fps_base;
    rc.idle_synth_fr = 0;
    rc.ncells = ncells;
    rc.prb = prb;
    rc.sym = sym;
    rc.frame_w = frame_w;
    rc.frame_h = frame_h;
    rc.off_x = off_x;
    rc.off_y = off_y;
    rc.content_w = content_w;
    rc.content_h = content_h;
    rc.mag_scale = mag_scale;
    rc.d_z = d_z[(size_t)i];
    rc.d_mag = d_mag[(size_t)i];
    rc.d_chan = d_chan[(size_t)i];
    rc.d_cell_rnti = d_cell_rnti[(size_t)i];
    rc.d_rgba = d_rgba[(size_t)i];
    rc.grid_mag = grid_mag;
    rc.block_mag = block_mag;
    rc.grid2 = grid2;
    rc.block2 = block2;
    rc.cuda_stream = (rt == 1) ? raster_cuda_stream : raster_streams[(size_t)i];
  }

  if (rt > 1) {
    raster_work_q = std::make_unique<RasterWorkQueue>(std::max(qdepth, rt * 2));
    raster_pool.rgba_out = rgba_q.get();
    raster_ap.work = raster_work_q.get();
    raster_ap.pool = &raster_pool;
    raster_ap.dm = dm;
    raster_ap.internal_q = internal_q.get();
    raster_ap.full_grid_q = full_grid_fifo_q.get();
    raster_ap.assembled_q = slot_assembled_q.get();
    raster_ap.sidecar_q = iq_sidecar_q.get();
    raster_ap.fps_base = fps_base;
    raster_ap.ncells = ncells;
    raster_ap.prb = prb;
    raster_ap.sym = sym;
    for (int i = 0; i < rt; ++i) {
      raster_ctxs[(size_t)i].internal_q = nullptr;
      raster_ctxs[(size_t)i].full_grid_q = nullptr;
      raster_ctxs[(size_t)i].assembled_q = nullptr;
      raster_ctxs[(size_t)i].sidecar_q = nullptr;
      raster_ctxs[(size_t)i].rgba_out = nullptr;
    }
  }

  AssemblyStageCtx as_ctx{};
  as_ctx.dm = dm;
  as_ctx.internal_q = internal_q.get();
  as_ctx.fps_base = fps_base;
  as_ctx.fps = fps;
  as_ctx.ncells = ncells;
  as_ctx.prb = prb;
  as_ctx.sym = sym;
  as_ctx.full_grid_q = full_grid_fifo_q.get();
  as_ctx.iq_fd_full_grid = iq_fd_full_grid;
  as_ctx.slot_assembled_q = slot_assembled_q.get();
  as_ctx.slot_fifo_path = &slot_fifo;
  as_ctx.iq_sidecar_q = iq_sidecar_q.get();
  as_ctx.iq_fd_sidecar = iq_fd_sidecar;

  RasterSupervisorCtx rsup{};
  rsup.rt = rt;
  rsup.ctxs = raster_ctxs.data();
  rsup.rgba_bytes = rgba_bytes;
  if (rt > 1) {
    rsup.work_q = raster_work_q.get();
    rsup.pool = &raster_pool;
    rsup.ap = &raster_ap;
  }

  CapturerCtx capturer_ctx{};
  capturer_ctx.out_root_dir = capturer_dir.empty() ? nullptr : &capturer_dir;
  capturer_ctx.fps_base = fps_base;
  capturer_ctx.frame_w = frame_w;
  capturer_ctx.frame_h = frame_h;
  capturer_ctx.frame_q = capturer_q.get();
  CapturerCtx* const capturer_ptr = capturer_ctx.out_root_dir ? &capturer_ctx : nullptr;
  for (int i = 0; i < rt; ++i) raster_ctxs[(size_t)i].capturer = capturer_ptr;
  raster_pool.capturer = capturer_ptr;

  std::thread stats_th(pipeline_stage_output_stats_main, fps_base, dm, capturer_ptr);

  EncodeStageCtx enc_ctx{};
  enc_ctx.rgba_q = rgba_q.get();
  enc_ctx.ff_pipe_fd = ff_pipe;
  enc_ctx.enc_pol = enc_pol;
  enc_ctx.dm = dm;
  enc_ctx.encoder_fps = encoder_fps;
  enc_ctx.fps = fps;
  enc_ctx.fps_base = fps_base;
  enc_ctx.rgba_npixels = rgba_npixels;
  enc_ctx.rgba_bytes = rgba_bytes;
  if (enc_pol == EncodePolicy::Mean) {
    enc_ctx.mean_acc_r.assign(rgba_npixels, 0.f);
    enc_ctx.mean_acc_g.assign(rgba_npixels, 0.f);
    enc_ctx.mean_acc_b.assign(rgba_npixels, 0.f);
    enc_ctx.mean_acc_a.assign(rgba_npixels, 0u);
  }

  std::thread assembly_stage_th(assembly_stage_main, &as_ctx);
  std::thread raster_stage_th(raster_stage_supervisor_main, &rsup);
  std::thread encode_stage_th(encode_stage_main, &enc_ctx);

  encode_stage_th.join();
  raster_stage_th.join();
  capturer_join_workers(&capturer_ctx);
  assembly_stage_th.join();

  if (stats_th.joinable()) stats_th.join();
  if (iq_fd_full_grid >= 0) close(iq_fd_full_grid);
  if (iq_fd_sidecar >= 0) close(iq_fd_sidecar);

  close(ff_pipe);
  if (ff_pid > 0) {
    kill(ff_pid, SIGTERM);
    waitpid(ff_pid, nullptr, 0);
  }

  if (rt == 1) {
    if (raster_cuda_stream) {
      cudaStreamSynchronize(raster_cuda_stream);
      cudaStreamDestroy(raster_cuda_stream);
      raster_cuda_stream = nullptr;
    }
  } else {
    for (int i = 0; i < rt; ++i) {
      cudaStreamSynchronize(raster_streams[(size_t)i]);
      cudaStreamDestroy(raster_streams[(size_t)i]);
    }
    raster_streams.clear();
  }

  free_raster_cuda_bufs();
  return 0;
}
