/**
 * Preset **`source`** values for **POST /api/generator/start** when **`mode`** is **`SYNTHETIC_IMAGES`**
 * (must match backend `_LAVFI_SOURCE_PRESETS` in `main.py`).
 */
export type SyntheticImagesLavfiPreset = {
  id: string;
  label: string;
  description: string;
};

export const LAVFI_PRESETS_FOR_SYNTHETIC_IMAGES: readonly SyntheticImagesLavfiPreset[] = [
  {
    id: "testsrc_720p30",
    label: "Test pattern 1280×720 @ 30",
    description: "Default-style bars; good for realtime NVENC.",
  },
  {
    id: "testsrc_480p30",
    label: "Test pattern 854×480 @ 30",
    description: "Lighter CPU load.",
  },
  {
    id: "smptebars_720p30",
    label: "SMPTE color bars 1280×720 @ 30",
    description: "Standard bars pattern.",
  },
  {
    id: "mandelbrot_960p30",
    label: "Mandelbrot 960×540 @ 30",
    description:
      "FFmpeg lavfi mandelbrot (960×540 @ 30), scaled to the plotter canvas via generator.py → RGBA FIFO (heavier than testsrc).",
  },
  {
    id: "randomnoise_960p30",
    label: "Random RGB noise (Python)",
    description:
      "Each frame: NumPy random uint8 RGB and opaque alpha at the plotter canvas size (no FFmpeg lavfi).",
  },
] as const;
