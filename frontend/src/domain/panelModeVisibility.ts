import {
  isIqCudaMode,
  normalizeUiMode,
  SYNTHETIC_FAPI,
  SYNTHETIC_IMAGES,
  SYNTHETIC_RESOURCE_GRID,
} from "./publisherModes";

/**
 * Object-centered panel layout: each column owns two objects — **Plotter** and **Generator** —
 * and each object has its own Configuration / Control / Status sub-sections.
 *
 * The **mode** dropdown lives on the **Generator** object, because the mode determines what
 * the generator produces; the plotter only consumes whatever IQ the generator emits.
 *
 * **Start generator / Stop generator** call **POST /api/generator/start** with the same **`mode`**
 * strings as the UI: **`SYNTHETIC_IMAGES`**, **`SYNTHETIC_RESOURCE_GRID`**, or **`SYNTHETIC_FAPI`**.
 * Lavfi presets apply only to **`SYNTHETIC_IMAGES`**.
 *
 * ## By UI mode (dropdown)
 *
 * ### `SYNTHETIC_IMAGES`
 * - **Plotter** · Config: **Encoder FPS (start)**. Control: Start/Stop plotter + **File capturer (JPEG)** (same for every **mode**; server **`capturer_capable`** still enables buttons). Status: MediaMTX path, **Publishing**, Plotter (CUDA), pipeline Hz rows + Published frames (metrics follow the running plotter / server **generation_mode**, not the Generator dropdown).
 * - **Generator** · Config: **Mode** + **Pattern** (video feed presets for **`SYNTHETIC_IMAGES`** only). Control: Start/Stop generator (all modes). Status: **Named pipe (UI mode)**. FAPI / slot writer row + **Generator FPS** row hidden.
 *
 * ### `SYNTHETIC_RESOURCE_GRID`
 * - **Plotter** · Same as **`SYNTHETIC_IMAGES`** (capturer is not hidden by Generator **mode**).
 * - **Generator** · Config: **Mode** + Generator FPS (no Pattern / lavfi row — that row is **`SYNTHETIC_IMAGES`** only). Control: Start/Stop generator (all modes). Status: **Named pipe (UI mode)**, FAPI / slot writer row + **Generator FPS** (same `<dl>`).
 *
 * ### `SYNTHETIC_FAPI`
 * - **Plotter** · Same as `SYNTHETIC_RESOURCE_GRID` (slot pipeline metrics in detail block).
 * - **Generator** · Config: **Mode** + Generator FPS (lavfi row hidden). Control: Start/Stop generator (all modes). Status: **Named pipe (UI mode)**, FAPI / slot writer row + **Generator FPS** (same `<dl>`).
 */
export const PANEL_VISIBILITY_BY_MODE = {
  [SYNTHETIC_IMAGES]: {
    plotter: {
      configuration: ["Encoder FPS (start)"],
      control: ["Plotter (CUDA) start/stop", "File capturer (JPEG)"],
      status: [
        "MediaMTX path",
        "Publishing",
        "Plotter (CUDA)",
        "Plotter detail (pipeline Hz rows, Published frames)",
      ],
    },
    generator: {
      configuration: ["Mode", "Pattern (`SYNTHETIC_IMAGES`)"],
      control: ["Start/Stop generator"],
      status: ["Named pipe (UI mode)"],
    },
  },
  [SYNTHETIC_RESOURCE_GRID]: {
    plotter: {
      configuration: ["Encoder FPS (start)"],
      control: ["Plotter (CUDA) start/stop", "File capturer (JPEG)"],
      status: [
        "MediaMTX path",
        "Publishing",
        "Plotter (CUDA)",
        "Plotter detail (pipeline Hz rows, Published frames)",
      ],
    },
    generator: {
      configuration: ["Mode", "Generator FPS"],
      control: ["Start/Stop generator"],
      status: [
        "Named pipe (UI mode)",
        "FAPI / slot writer status",
        "Generator FPS",
      ],
    },
  },
  [SYNTHETIC_FAPI]: {
    plotter: {
      configuration: ["Encoder FPS (start)"],
      control: ["Plotter (CUDA) start/stop", "File capturer (JPEG)"],
      status: [
        "MediaMTX path",
        "Publishing",
        "Plotter (CUDA)",
        "Plotter detail (pipeline Hz rows, Published frames)",
      ],
    },
    generator: {
      configuration: ["Mode", "Generator FPS"],
      control: ["Start/Stop generator"],
      status: [
        "Named pipe (UI mode)",
        "FAPI / slot writer status",
        "Generator FPS",
      ],
    },
  },
} as const;

export type PanelModeVisibility = {
  showStatusIqFeedGeneratorRow: boolean;
  /** Generator FPS row group in the Generator status `<dl>`; not used for Plotter pipeline rows (those stay visible). */
  showStatusIqDetailBlock: boolean;
  showControlPlotter: boolean;
  showConfigGeneratorFps: boolean;
};

export function panelVisibilityForMode(uiModeRaw: string): PanelModeVisibility {
  const mode = normalizeUiMode(uiModeRaw);
  const video = mode === SYNTHETIC_IMAGES;
  const iq = isIqCudaMode(mode);
  if (video) {
    return {
      showStatusIqFeedGeneratorRow: false,
      showStatusIqDetailBlock: false,
      showControlPlotter: true,
      showConfigGeneratorFps: false,
    };
  }
  if (iq) {
    return {
      showStatusIqFeedGeneratorRow: true,
      showStatusIqDetailBlock: true,
      showControlPlotter: true,
      showConfigGeneratorFps: true,
    };
  }
  return {
    showStatusIqFeedGeneratorRow: false,
    showStatusIqDetailBlock: false,
    showControlPlotter: true,
    showConfigGeneratorFps: false,
  };
}
