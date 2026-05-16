export type GenerationStats = {
  fps: number | null;
  frame: number | null;
  speed?: number | null;
  bitrate_kbits?: number | null;
};

/**
 * Merged view of **GET /api/plotter/status** plus **GET /api/generator/status** fields the UI needs
 * (plotter metrics + generator mode / mq slices).
 */
export type PublisherStatus = {
  /** Backend **`generation_mode`** (process **GENERATION_MODE**), mapped to **`mode`** for the UI. */
  mode: string;
  generating: boolean;
  /** True while resgrid_plotter is running (authoritative for Stop plotter). */
  cuda_plotter_active?: boolean;
  video_encoder: string;
  lavfi_video: string | null;
  /** Preset id from `source` when the video generator (FFmpeg or RGBA FIFO feeder) is active; null for env default / custom graph. */
  video_generator_source?: string | null;
  /** True while generator.py (RGBA mq) or legacy FFmpeg lavfi is publishing; independent of `video_generator_source`. */
  video_generator_running?: boolean;
  stats: GenerationStats | null;
  publish_rtsp: string;
  mtx_path: string;
  iq_prb?: number;
  iq_symbols?: number;
  /**
   * Inclusive 0-based first OFDM-symbol index the plotter renders
   * (**START_SYMBOL** env / **--start-symbol** CLI). Defaults to **0** = full slot.
   */
  iq_symbol_start?: number;
  /**
   * Inclusive 0-based last OFDM-symbol index the plotter renders
   * (**STOP_SYMBOL** env / **--stop-symbol** CLI). Defaults to **iq_symbols - 1** = full slot.
   */
  iq_symbol_stop?: number;
  iq_producer_fps?: number | null;
  iq_plotter_fps?: number | null;
  iq_producer_command_fps?: number | null;
  iq_generator_running?: boolean;
  iq_generator_capable?: boolean;
  /** From **GET /api/generator/status** `synthetic_video` (POSIX mq name for RGBA **generator.py**). */
  video_generator_mq_path?: string | null;
  /** Resource-grid IQ mq writer slice (server **synthetic_resource_grid**). */
  synthetic_resource_grid?: {
    running: boolean;
    capable: boolean;
    /** **generator.py** OS pid when **running** (from server). */
    pid?: number | null;
    mq_path?: string | null;
  };
  /** Slot / FAPI mq writer slice (server **synthetic_fapi**). */
  synthetic_fapi?: {
    running: boolean;
    capable: boolean;
    pid?: number | null;
    mq_path?: string | null;
  };
  /** CUDA SHM @40 Hz */
  assembly_input_msg_s?: number | null;
  /** CUDA SHM @48 Hz */
  assembly_output_slot_s?: number | null;
  /** CUDA SHM @56 — same as assembly_output_slot_s (chained “Raster input”). */
  raster_input_slot_s?: number | null;
  /** CUDA SHM @64 (same as @8) */
  raster_output_image_s?: number | null;
  /** CUDA SHM @72 — same as raster_output_image_s (chained “Encoder input”). */
  encoder_input_image_s?: number | null;
  /** CUDA SHM @80 Hz */
  encoder_output_fps?: number | null;
  /** File capturer (JPEG) enabled; parallel to **encoder** (RTSP/video). */
  capturer_capable?: boolean;
  capturer_active?: boolean | null;
  capturer_phase?: string | null;
  capturer_inflight?: number | null;
  capturer_written?: number | null;
  capturer_total?: number | null;
  capturer_out_dir?: string | null;
  capturer_queue_depth?: number | null;
  capturer_queue_cap?: number | null;
  capturer_jpeg_threads?: number | null;
};

export type IqChannelLegendPayload = {
  channels: {
    id: number;
    label: string;
    rgb: { r: number; g: number; b: number };
    hex: string;
  }[];
  channel_zero_note: string;
  render_note: string;
};

export type PlaybackUrls = {
  hls: string;
  webrtc_whep: string;
};
