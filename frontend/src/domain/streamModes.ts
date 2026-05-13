import type { StreamStatus } from "./streamTypes";

/** Backend STREAM_MODE strings for UI preset radios (1…3). */
export const STREAM_MODE_SYNTHETIC_VIDEO = "SYNTHETIC_VIDEO";
export const STREAM_MODE_SYNTHETIC_IQ = "SYNTHETIC_IQ";
export const STREAM_MODE_SYNTHETIC_CHANNEL_BASED = "SYNTHETIC_CHANNEL_BASED";
export const LEGACY_STREAM_MODE_CHANNEL_BASED = "CHANNEL_BASED_SYNTHETIC_GENERATION";

export const PUBLISHER_MODES: readonly { num: number; mode: string }[] = [
  { num: 1, mode: STREAM_MODE_SYNTHETIC_VIDEO },
  { num: 2, mode: STREAM_MODE_SYNTHETIC_IQ },
  { num: 3, mode: STREAM_MODE_SYNTHETIC_CHANNEL_BASED },
];

export function isIqCudaStreamMode(mode: string): boolean {
  return (
    mode === STREAM_MODE_SYNTHETIC_IQ ||
    mode === STREAM_MODE_SYNTHETIC_CHANNEL_BASED ||
    mode === LEGACY_STREAM_MODE_CHANNEL_BASED
  );
}

export function isChannelBasedIqMode(mode: string): boolean {
  return (
    mode === STREAM_MODE_SYNTHETIC_CHANNEL_BASED ||
    mode === LEGACY_STREAM_MODE_CHANNEL_BASED
  );
}

/** Synthetic IQ pattern (Viridis demo) applies to full-grid FIFO only, not channel-based mode. */
export function showsSyntheticIqPatternUi(mode: string): boolean {
  return isIqCudaStreamMode(mode) && !isChannelBasedIqMode(mode);
}

/** True when API reports CUDA IQ pipeline is up (resgrid_plotter). */
export function cudaPlotterRunningFromStatus(s: StreamStatus): boolean {
  return (
    isIqCudaStreamMode(s.stream_mode) &&
    s.generating &&
    s.video_encoder === "cuda_iq_pipeline"
  );
}

/** True when API reports FFmpeg synthetic video is up. */
export function ffmpegPublisherRunningFromStatus(s: StreamStatus): boolean {
  return s.stream_mode === STREAM_MODE_SYNTHETIC_VIDEO && s.generating;
}

/** Map legacy API mode to the radio value shown in the UI. */
export function normalizeUiStreamMode(streamMode: string): string {
  return streamMode === LEGACY_STREAM_MODE_CHANNEL_BASED
    ? STREAM_MODE_SYNTHETIC_CHANNEL_BASED
    : streamMode;
}
