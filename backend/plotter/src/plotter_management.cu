/*
 * Management: process globals, queue-drop warnings, CLI parsing, and `plotter_run_launch` (see `plotter_pipeline.cu`).
 */
#include "plotter_pipeline.cuh"
#include "plotter_posix_mq.cuh"

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

/** Default POSIX mq name for raw RGBA (matches ``RESGRID_MQ_IMAGES`` default in FastAPI ``main.py``). */
static constexpr const char kDefaultMqImagesPath[] = "/images";
/** Default full-grid I/Q mq (matches ``CUDA_IQ_MQ_PATH`` default in ``main.py``). */
static constexpr const char kDefaultMqResourceGridPath[] = "/resource_grid";
/** Default slot / FAPI mq (matches ``CUDA_SLOT_MQ_PATH`` default in ``main.py``). */
static constexpr const char kDefaultMqFapiPath[] = "/fapi";

static void usage(const char* av0) {
  fprintf(stderr,
          "Usage: %s [--rtsp URL] [--prb N] [--symbols M] [--start-symbol S0] [--stop-symbol S1] [--cell-pixels P] "
          "[--fps F] [--encoder-fps E] [--mag-scale S] "
          "[--preset p4] [--video-encoder nvenc|cpu] [--queue-depth D] [--raster-threads N] [--encode-policy POL] "
          "[--no-square] [--square-letterbox] "
          "[--mq-images [NAME]] [--mq-resource-grid [NAME]] [--mq-resource-grid-mode full|slot] [--mq-fapi [NAME]] "
          "[--cuda-shm NAME] [--capturer-dir DIR]\n"
          "  --start-symbol S0  Inclusive 0-based first OFDM symbol to render (default 0).\n"
          "  --stop-symbol S1   Inclusive 0-based last OFDM symbol to render (default --symbols-1).\n"
          "                     Combine to crop the raster horizontally to columns [S0, S1]; producers\n"
          "                     still publish the full slot, only the visualization is narrowed.\n"
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
          "                     `--mq-fapi` is used, idle encoder ticks do not inject synthetic black frames.\n"
          "  Input (POSIX message queues; names start with ``/``; chunked wire — see ``readme.md``; no numeric data mode):\n"
          "    --mq-images [NAME]  Raw RGBA to the encoder (optional alongside IQ queues: muxed with raster RGBA\n"
          "                        on FFmpeg stdin under one lock). If NAME omitted, default is %s (backend RESGRID_MQ_IMAGES).\n"
          "                        With IQ queues only: full raster + assembler + encoder pipeline; without IQ queues:\n"
          "                        FFmpeg-only fast path.\n"
          "    --mq-resource-grid [NAME]  With `--mq-fapi`: raw full-grid I/Q sidecar (one logical frame = one raster).\n"
          "                        Without `--mq-fapi`: ``full`` (default) = one raw full-grid raster per logical frame;\n"
          "                        ``slot`` = framed slot-IQ stream (header + complex patch per message, see\n"
          "                        ``ResourceGridSlotIqHdr`` in ``plotter_common.cuh``) → same CUDA color path as slots.\n"
          "                        If NAME omitted, default is %s (same as backend CUDA_IQ_MQ_PATH).\n"
          "    --mq-resource-grid-mode full|slot  Wire format for standalone ``--mq-resource-grid`` (ignored with\n"
          "                        ``--mq-fapi``). Default ``full``.\n"
          "    --mq-fapi [NAME]  Slot / FAPI messages → slot assembler → `AssembledSlotQueue` (PHY patches + metadata).\n"
          "                        If NAME is omitted, default is %s (same as backend CUDA_SLOT_MQ_PATH).\n"
          "    (no IQ queues)     Internal synthetic I/Q producer at `--fps`.\n"
          "  --raster-threads N  CUDA raster workers (default 16). N>1: one acquirer preserves slot/sidecar order, N\n"
          "                     GPU workers + merger reorder RGBA for the encoder (cap 64).\n"
          "  --queue-depth D    In-process GridFrame/RgbaFrame queue capacity (default 32); backend env CUDA_IQ_QUEUE_DEPTH.\n",
          av0, kDefaultMqImagesPath, kDefaultMqResourceGridPath, kDefaultMqFapiPath);
}

int main(int argc, char** argv) {
  std::string rtsp = "rtsp://127.0.0.1:8554/live";
  int prb = 273;
  int sym = 14;
  int sym_start_cli = -1;
  int sym_stop_cli = -1;
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
  std::string mq_images;
  std::string mq_resource_grid;
  std::string mq_fapi;
  std::string mq_resource_grid_mode = "full";
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
    else if (!strcmp(argv[i], "--start-symbol") && i + 1 < argc)
      sym_start_cli = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stop-symbol") && i + 1 < argc)
      sym_stop_cli = atoi(argv[++i]);
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
    else if (!strcmp(argv[i], "--mq-images")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        mq_images = argv[++i];
      else
        mq_images = kDefaultMqImagesPath;
    } else if (!strcmp(argv[i], "--mq-resource-grid")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        mq_resource_grid = argv[++i];
      else
        mq_resource_grid = kDefaultMqResourceGridPath;
    } else if (!strcmp(argv[i], "--mq-resource-grid-mode") && i + 1 < argc)
      mq_resource_grid_mode = argv[++i];
    else if (!strcmp(argv[i], "--mq-fapi")) {
      if (i + 1 < argc && argv[i + 1][0] != '-')
        mq_fapi = argv[++i];
      else
        mq_fapi = kDefaultMqFapiPath;
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

  /* Symbol render window (inclusive). Unset → full slot; out-of-range / inverted → clamp + warn so a
   * misconfigured env still produces a usable raster. */
  int sym_disp_start = (sym_start_cli >= 0) ? sym_start_cli : 0;
  int sym_disp_stop = (sym_stop_cli >= 0) ? sym_stop_cli : (sym - 1);
  if (sym_disp_start < 0 || sym_disp_start >= sym) {
    fprintf(stderr, "%s: warning: --start-symbol %d out of range [0,%d]; using 0\n", argv[0], sym_disp_start, sym - 1);
    sym_disp_start = 0;
  }
  if (sym_disp_stop < 0 || sym_disp_stop >= sym) {
    fprintf(stderr, "%s: warning: --stop-symbol %d out of range [0,%d]; using %d\n", argv[0], sym_disp_stop, sym - 1,
            sym - 1);
    sym_disp_stop = sym - 1;
  }
  if (sym_disp_stop < sym_disp_start) {
    fprintf(stderr, "%s: warning: --stop-symbol %d < --start-symbol %d; rendering full slot\n", argv[0],
            sym_disp_stop, sym_disp_start);
    sym_disp_start = 0;
    sym_disp_stop = sym - 1;
  }
  int sym_disp_count = sym_disp_stop - sym_disp_start + 1;

  const bool have_fapi = !mq_fapi.empty();
  const bool have_res = !mq_resource_grid.empty();
  if (!str_ieq(mq_resource_grid_mode.c_str(), "full") && !str_ieq(mq_resource_grid_mode.c_str(), "slot")) {
    fprintf(stderr, "%s: --mq-resource-grid-mode must be full or slot (got %s)\n", argv[0],
            mq_resource_grid_mode.c_str());
    return 1;
  }
  if (have_fapi && str_ieq(mq_resource_grid_mode.c_str(), "slot")) {
    fprintf(stderr,
            "%s: --mq-resource-grid-mode slot applies only without --mq-fapi (FAPI sidecar expects raw full-grid "
            "writes)\n",
            argv[0]);
    return 1;
  }
  const bool resource_grid_slot_stream =
      have_res && !have_fapi && str_ieq(mq_resource_grid_mode.c_str(), "slot");

  /* Size the canvas from the first live header so CUDA_IQ_PRB does not have to match the cell. */
  if (have_res && !resource_grid_slot_stream) {
    const bool zmq_iq = mq_resource_grid.rfind("ipc://", 0) == 0 || mq_resource_grid.rfind("tcp://", 0) == 0 ||
                        mq_resource_grid.rfind("inproc://", 0) == 0;
    int det_prb = 0;
    int det_sym = 0;
    if (zmq_iq && resgrid_zmq_peek_grid_dims(mq_resource_grid.c_str(), 5000, &det_prb, &det_sym)) {
      fprintf(stderr, "%s: grid size from ZeroMQ header prb=%d symbols=%d\n", argv[0], det_prb, det_sym);
      prb = det_prb;
      sym = det_sym;
      sym_disp_start = (sym_start_cli >= 0) ? sym_start_cli : 0;
      sym_disp_stop = (sym_stop_cli >= 0) ? sym_stop_cli : (sym - 1);
      if (sym_disp_start < 0 || sym_disp_start >= sym) sym_disp_start = 0;
      if (sym_disp_stop < 0 || sym_disp_stop >= sym) sym_disp_stop = sym - 1;
      if (sym_disp_stop < sym_disp_start) {
        sym_disp_start = 0;
        sym_disp_stop = sym - 1;
      }
      sym_disp_count = sym_disp_stop - sym_disp_start + 1;
    } else if (zmq_iq) {
      fprintf(stderr, "%s: no ZeroMQ resource-grid header within 5s; using --prb %d --symbols %d\n", argv[0], prb,
              sym);
    }
  }

  int frame_w;
  int frame_h;
  int off_x = 0;
  int off_y = 0;
  int content_w;
  int content_h;
  float cs_x;
  float cs_y;

  /* Canvas geometry uses the *visible* symbol count so cropping with --start-symbol/--stop-symbol
   * narrows the raster horizontally instead of leaving black margins. */
  if (square_canvas) {
    int side = std::max(prb * cell_px, sym_disp_count * cell_px);
    if (side % 2) ++side;
    frame_w = frame_h = side;
    if (square_letterbox) {
      const float cell_scale = (float)side / (float)std::max(prb, sym_disp_count);
      content_w = (int)lrintf((float)sym_disp_count * cell_scale);
      content_h = (int)lrintf((float)prb * cell_scale);
      off_x = (frame_w - content_w) / 2;
      off_y = (frame_h - content_h) / 2;
      cs_x = cs_y = cell_scale;
    } else {
      off_x = off_y = 0;
      content_w = content_h = side;
      cs_x = (float)side / (float)sym_disp_count;
      cs_y = (float)side / (float)prb;
    }
  } else {
    cs_x = cs_y = (float)cell_px;
    content_w = sym_disp_count * cell_px;
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
  /* RGBA-only fast path only when there are no IQ inputs; otherwise **mq_images** feeds the encoder in parallel
   * with the raster queue (see **EncoderStage** dual-input threads in **plotter_encoder.cu**). */
  lc.mq_images_only = !mq_images.empty() && mq_resource_grid.empty() && mq_fapi.empty();
  lc.rtsp = std::move(rtsp);
  lc.cuda_shm_name = std::move(cuda_shm_name);
  lc.mq_images = std::move(mq_images);
  lc.mq_resource_grid = std::move(mq_resource_grid);
  lc.mq_fapi = std::move(mq_fapi);
  lc.capturer_dir = std::move(capturer_dir);
  lc.preset = preset;
  lc.prb = prb;
  lc.sym = sym;
  lc.sym_start = sym_disp_start;
  lc.sym_count = sym_disp_count;
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
