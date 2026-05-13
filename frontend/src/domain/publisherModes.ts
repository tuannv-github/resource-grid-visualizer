import type { PublisherStatus } from "./publisherStatus";
import { SYNTHETIC_FAPI, SYNTHETIC_IMAGES, SYNTHETIC_RESOURCE_GRID } from "./generatorModes";

export { SYNTHETIC_FAPI, SYNTHETIC_IMAGES, SYNTHETIC_RESOURCE_GRID };

/** UI / process **GENERATION_MODE** (same strings as **GET /api/generator/status** `generation_mode` and generator **mode**). */
export const PUBLISHER_MODE_PRESETS: readonly { num: number; mode: string }[] = [
  { num: 1, mode: SYNTHETIC_IMAGES },
  { num: 2, mode: SYNTHETIC_RESOURCE_GRID },
  { num: 3, mode: SYNTHETIC_FAPI },
];

export function isIqCudaMode(mode: string): boolean {
  const m = normalizeUiMode(mode);
  return m === SYNTHETIC_RESOURCE_GRID || m === SYNTHETIC_FAPI;
}

/** True when the CUDA plotter process is running (uses **cuda_plotter_active** from the API when present). */
export function cudaPlotterRunningFromStatus(s: PublisherStatus): boolean {
  if (s.cuda_plotter_active === true) return true;
  if (s.cuda_plotter_active === false) return false;
  return s.generating === true && s.video_encoder === "cuda_iq_pipeline";
}

/** Pipeline / IQ-axis metrics apply (IQ modes, or SYNTHETIC_IMAGES while the CUDA plotter is active). */
export function cudaHeatmapStatusRelevant(s: PublisherStatus): boolean {
  return isIqCudaMode(s.mode) || cudaPlotterRunningFromStatus(s);
}

/**
 * True when the **SYNTHETIC_IMAGES** feeder path is active: FFmpeg lavfi (legacy) or **generator.py** on the RGBA FIFO.
 *
 * Uses the backend **`video_generator_running`** flag when present. Otherwise infers from **`video_generator_source`**
 * for older responses that omit the boolean.
 */
export function syntheticImagesFeedRunningFromStatus(s: PublisherStatus): boolean {
  if (s.video_generator_running === true) return true;
  if (s.video_generator_running === false) return false;
  return s.video_generator_source != null && s.video_generator_source !== "";
}

/** Normalize historic **generation_mode** tokens to **SYNTHETIC_IMAGES** | **SYNTHETIC_RESOURCE_GRID** | **SYNTHETIC_FAPI**. */
export function normalizeUiMode(rawMode: string): string {
  const r = rawMode.trim();
  if (r === "SYNTHETIC_VIDEO") return SYNTHETIC_IMAGES;
  if (r === "SYNTHETIC_IQ") return SYNTHETIC_RESOURCE_GRID;
  if (r === "SYNTHETIC_CHANNEL_BASED") return SYNTHETIC_FAPI;
  return r;
}
