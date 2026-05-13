/*
 * Management: process globals, queue-drop warnings, CLI parsing, and `plotter_run_launch` (see `plotter_pipeline.cu`).
 */
#include "plotter_pipeline.cuh"

#include <algorithm>
#include <mutex>

std::atomic<bool> g_run{true};
std::atomic<uint64_t> g_producer_frames{0};
std::atomic<uint64_t> g_stage_output_count_assembler{0};
std::atomic<uint64_t> g_stage_output_count_raster{0};
std::atomic<uint64_t> g_stage_output_count_encoder{0};
std::atomic<uint64_t> g_slot_completed_drops{0};
std::atomic<uint64_t> g_grid_frame_queue_drops{0};
std::atomic<uint64_t> g_rgba_frame_queue_drops{0};
std::atomic<uint64_t> g_raster_work_queue_drops{0};
std::atomic<uint64_t> g_capturer_frame_queue_drops{0};
std::atomic<uint64_t> g_assembler_input_msgs{0};

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

/** Default raw RGBA named pipe (matches ``RESGRID_FIFO_IMAGES`` default in FastAPI ``main.py``). */
static constexpr const char kDefaultFifoImagesPath[] = "/tmp/images.fifo";
/** Default full-grid I/Q named pipe (matches ``CUDA_IQ_FIFO_PATH`` default in FastAPI ``main.py``). */
static constexpr const char kDefaultFifoResourceGridPath[] = "/tmp/resource_grid.fifo";
/** Default slot / FAPI named pipe (matches ``CUDA_SLOT_FIFO_PATH`` default in FastAPI ``main.py``). */
static constexpr const char kDefaultFifoFapiPath[] = "/tmp/fapi.fifo";

static void usage(const char* av0) {
  fprintf(stderr,
          "Usage: %s [--rtsp URL] [--prb N] [--symbols M] [--cell-pixels P] [--fps F] [--encoder-fps E] [--mag-scale S] "
          "[--preset p4] [--video-encoder nvenc|cpu] [--queue-depth D] [--raster-threads N] [--encode-policy POL] "
          "[--no-square] [--square-letterbox] "
          "[--fifo-images [PATH]] [--fifo-resource-grid [PATH]] [--fifo-resource-grid-mode full|slot] [--fifo-fapi [PATH]] "
          "[--cuda-shm NAME] [--capturer-dir DIR]\n"
          "  --fps F            Pipeline / internal synthetic producer default (Hz); default 200. Also sets --encoder-fps\n"
          "                     if you omit --encoder-fps (backward compatible single-knob).\n"
          "  --encoder-fps E    Target NVENC output rate (Hz); default 24. With --encode-policy last|mean, rawvideo -r\n"
          "                     matches E and the plotter paces stdin. With ffmpeg, rawvideo -r matches --fps (capped 600)\n"
          "                     and FFmpeg -vf fps=E reduces before NVENC.\n"
          "  --video-encoder nvenc|cpu  H.264 via FFmpeg h264_nvenc (default) or libx264 (no NVENC hardware).\n"
          "  --encode-policy POL  last (default): each encoder tick sends latest RGBA (subsample in time).\n"
          "                     mean: each tick sends per-pixel color = mean in linear RGB (sRGB EOTF), then OETF;\n"
          "                     alpha = rounded mean of input alphas.\n"
          "                     ffmpeg: write every rendered frame to stdin; FFmpeg fps filter → encoder-fps. When\n"
          "                     `--fifo-fapi` is used, idle encoder ticks do not inject synthetic black frames.\n"
          "  Input (combine any compatible FIFOs; no numeric data mode):\n"
          "    --fifo-images [PATH]  Raw RGBA to the encoder (optional alongside IQ FIFOs: muxed with raster RGBA\n"
          "                        on FFmpeg stdin under one lock). If PATH omitted, default is %s (backend RESGRID_FIFO_IMAGES).\n"
          "                        With IQ FIFOs only: full raster + assembler + encoder pipeline; without IQ FIFOs:\n"
          "                        FFmpeg-only fast path.\n"
          "    --fifo-resource-grid [PATH]  With `--fifo-fapi`: raw full-grid I/Q sidecar (one read = one raster).\n"
          "                        Without `--fifo-fapi`: ``full`` (default) = one raw full-grid raster per read;\n"
          "                        ``slot`` = framed slot-IQ stream (header + complex patch per message, see\n"
          "                        ``ResourceGridSlotIqHdr`` in ``plotter_common.cuh``) → same CUDA color path as slots.\n"
          "                        If PATH omitted, default is %s (same as backend CUDA_IQ_FIFO_PATH).\n"
          "    --fifo-resource-grid-mode full|slot  Wire format for standalone ``--fifo-resource-grid`` (ignored with\n"
          "                        ``--fifo-fapi``). Default ``full``.\n"
          "    --fifo-fapi [PATH]  Slot / FAPI messages → slot assembler → `AssembledSlotQueue` (PHY patches + metadata).\n"
          "                        If PATH is omitted, default is %s (same as backend CUDA_SLOT_FIFO_PATH).\n"
          "    (no IQ FIFOs)      Internal synthetic I/Q producer at `--fps`.\n"
          "  --raster-threads N  CUDA raster workers (default 16). N>1: one acquirer preserves slot/sidecar order, N\n"
          "                     GPU workers + merger reorder RGBA for the encoder (cap 64).\n"
          "  --queue-depth D    In-process GridFrame/RgbaFrame queue capacity (default 32); backend env CUDA_IQ_QUEUE_DEPTH.\n",
          av0, kDefaultFifoImagesPath, kDefaultFifoResourceGridPath, kDefaultFifoFapiPath);
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
  std::string fifo_images;
  std::string fifo_resource_grid;
  std::string fifo_fapi;
  std::string fifo_resource_grid_mode = "full";
  std::string capturer_dir;
  std::string cuda_shm_name;
  std::string encode_policy_arg = "last";
  bool video_encoder_nvenc = true;

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
    else if (!strcmp(argv[i], "--fifo-images")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        fifo_images = argv[++i];
      else
        fifo_images = kDefaultFifoImagesPath;
    } else if (!strcmp(argv[i], "--fifo-resource-grid")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        fifo_resource_grid = argv[++i];
      else
        fifo_resource_grid = kDefaultFifoResourceGridPath;
    }
    else if (!strcmp(argv[i], "--fifo-resource-grid-mode") && i + 1 < argc)
      fifo_resource_grid_mode = argv[++i];
    else if (!strcmp(argv[i], "--fifo-fapi")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        fifo_fapi = argv[++i];
      else
        fifo_fapi = kDefaultFifoFapiPath;
    }
    else if (!strcmp(argv[i], "--cuda-shm") && i + 1 < argc)
      cuda_shm_name = argv[++i];
    else if (!strcmp(argv[i], "--capturer-dir") && i + 1 < argc)
      capturer_dir = argv[++i];
    else if (!strcmp(argv[i], "--encoder-snap-dir") && i + 1 < argc)
      capturer_dir = argv[++i];
    else if (!strcmp(argv[i], "--encode-policy") && i + 1 < argc)
      encode_policy_arg = argv[++i];
    else if (!strcmp(argv[i], "--video-encoder") && i + 1 < argc) {
      const char* ve = argv[++i];
      if (str_ieq(ve, "cpu"))
        video_encoder_nvenc = false;
      else if (str_ieq(ve, "nvenc"))
        video_encoder_nvenc = true;
      else {
        fprintf(stderr, "%s: unknown --video-encoder %s (use nvenc|cpu)\n", argv[0], ve);
        return 1;
      }
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
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

  const bool have_fapi = !fifo_fapi.empty();
  const bool have_res = !fifo_resource_grid.empty();
  if (!str_ieq(fifo_resource_grid_mode.c_str(), "full") && !str_ieq(fifo_resource_grid_mode.c_str(), "slot")) {
    fprintf(stderr, "%s: --fifo-resource-grid-mode must be full or slot (got %s)\n", argv[0],
            fifo_resource_grid_mode.c_str());
    return 1;
  }
  if (have_fapi && str_ieq(fifo_resource_grid_mode.c_str(), "slot")) {
    fprintf(stderr,
            "%s: --fifo-resource-grid-mode slot applies only without --fifo-fapi (FAPI sidecar expects raw full-grid "
            "writes)\n",
            argv[0]);
    return 1;
  }
  const bool resource_grid_slot_stream =
      have_res && !have_fapi && str_ieq(fifo_resource_grid_mode.c_str(), "slot");

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

  PlotterLaunchConfig lc{};
  lc.prog = argv[0];
  /* RGBA-only fast path only when there are no IQ inputs; otherwise **fifo_images** feeds the encoder in parallel
   * with the raster queue (see **EncoderStage** dual-input threads in **plotter_encoder.cu**). */
  lc.fifo_images_only = !fifo_images.empty() && fifo_resource_grid.empty() && fifo_fapi.empty();
  lc.rtsp = std::move(rtsp);
  lc.cuda_shm_name = std::move(cuda_shm_name);
  lc.fifo_images = std::move(fifo_images);
  lc.fifo_resource_grid = std::move(fifo_resource_grid);
  lc.fifo_fapi = std::move(fifo_fapi);
  lc.capturer_dir = std::move(capturer_dir);
  lc.preset = preset;
  lc.prb = prb;
  lc.sym = sym;
  lc.fps = fps;
  lc.encoder_fps = encoder_fps;
  lc.qdepth = qdepth;
  lc.raster_threads = raster_threads;
  lc.mag_scale = mag_scale;
  lc.enc_pol = enc_pol;
  lc.video_encoder_nvenc = video_encoder_nvenc;
  lc.square_canvas = square_canvas;
  lc.square_letterbox = square_letterbox;
  lc.have_fapi = have_fapi;
  lc.have_res = have_res;
  lc.resource_grid_slot_stream = resource_grid_slot_stream;
  lc.frame_w = frame_w;
  lc.frame_h = frame_h;
  lc.off_x = off_x;
  lc.off_y = off_y;
  lc.content_w = content_w;
  lc.content_h = content_h;
  lc.cs_x = cs_x;
  lc.cs_y = cs_y;
  return plotter_run_launch(lc);
}
