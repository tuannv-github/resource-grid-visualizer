export type GenerationStats = {
  fps: number | null;
  frame: number | null;
  speed?: number | null;
  bitrate_kbits?: number | null;
};

export type StreamStatus = {
  stream_mode: string;
  generating: boolean;
  video_encoder: string;
  lavfi_video: string | null;
  stats: GenerationStats | null;
  publish_rtsp: string;
  mtx_path: string;
  iq_prb?: number;
  iq_symbols?: number;
  iq_producer_fps?: number | null;
  iq_plotter_fps?: number | null;
  iq_producer_command_fps?: number | null;
  iq_pattern?: number | null;
  iq_generator_running?: boolean;
  iq_generator_capable?: boolean;
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

export type IqPatternCatalog = {
  patterns: { id: number; label: string }[];
};

export type PlaybackUrls = {
  hls: string;
  webrtc_whep: string;
};
