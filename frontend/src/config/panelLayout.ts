import { envTruthy } from "./env";

export type PanelBoot = {
  /** DOM root for video + IQ legend (left column). */
  id: string;
  /** DOM root for status, configuration, and actions (right column, inside `<details>`). */
  controlsRootId: string;
  apiBase: string;
  playbackPath: string;
  /** Distinct HTML **id** for each panel’s mode `<select>`. */
  modeSelectId: string;
  heading: string;
};

export type GuiLayoutMode = "combine" | "dlul" | "rxnai" | "ulnai" | "auto";

export type ResolvedGuiLayout = {
  dual: boolean;
  panels: PanelBoot[];
  layoutMode: GuiLayoutMode;
};

/**
 * Primary GUI layout selector (Vite build-time).
 * **`combine`** — one column, single API (`VITE_API_BASE` or relative proxy).
 * **`dlul`** — Downlink + Uplink (`VITE_API_BASE_DL` + `VITE_API_BASE_UL`).
 * **`rxnai`** — RX + NAI (`VITE_API_BASE_RX` + `VITE_API_BASE_NAI`; no DL panels).
 * **`ulnai`** — deprecated alias for **`rxnai`**.
 * **`auto`** — infer from API bases (rx+nai → rxnai, dl+ul → dlul, else combine).
 */
export type GuiModeEnv = "combine" | "dlul" | "rxnai" | "ulnai" | "auto";

function parseGuiModeEnv(raw: string | undefined): GuiModeEnv | null {
  const t = raw?.trim().toLowerCase();
  if (!t) return null;
  if (t === "combine" || t === "single" || t === "one") return "combine";
  if (t === "dlul" || t === "dl_ul" || t === "dl-ul") return "dlul";
  if (t === "rxnai" || t === "rx_nai" || t === "rx-nai") return "rxnai";
  if (t === "ulnai" || t === "ul_nai" || t === "ul-nai") return "rxnai";
  if (t === "auto") return "auto";
  console.warn(
    `[resgrid] Unknown VITE_GUI_MODE="${raw}"; use combine | dlul | rxnai | auto`,
  );
  return null;
}

/** Resolve layout mode: `VITE_GUI_MODE` wins; legacy `VITE_GUI_*` flags are fallbacks. */
function resolveGuiModeEnv(): GuiModeEnv {
  const explicit = parseGuiModeEnv(import.meta.env.VITE_GUI_MODE);
  if (explicit) return explicit;

  if (envTruthy(import.meta.env.VITE_GUI_COMBINE)) return "combine";
  if (envTruthy(import.meta.env.VITE_GUI_RXNAI)) return "rxnai";
  if (envTruthy(import.meta.env.VITE_GUI_ULNAI)) return "rxnai";
  if (envTruthy(import.meta.env.VITE_GUI_DLUL)) return "dlul";

  return "auto";
}

/**
 * If the UI is opened from a LAN host (e.g. http://10.1.101.34:5173) but VITE_* still points at
 * localhost/127.0.0.1, rewrite the API hostname so /docs and fetch() hit the same machine as the page.
 */
function rewriteLoopbackApiBaseToPageHost(apiBase: string): string {
  const base = apiBase.trim().replace(/\/$/, "");
  if (typeof window === "undefined") return base;
  try {
    const u = new URL(base);
    const loopback = u.hostname === "localhost" || u.hostname === "127.0.0.1";
    const h = window.location.hostname;
    if (loopback && h && h !== "localhost" && h !== "127.0.0.1") {
      u.hostname = h;
      return u.toString().replace(/\/$/, "");
    }
  } catch {
    /* leave base unchanged */
  }
  return base;
}

function singleApiBase(): string {
  /** With Vite dev proxy (Docker), call the API on the same origin as the UI (:5173) so LAN users need not open :8081. */
  if (envTruthy(import.meta.env.VITE_API_RELATIVE)) return "";
  const single = import.meta.env.VITE_API_BASE?.trim();
  const raw =
    single ||
    (typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}:8081`
      : "http://localhost:8081");
  return rewriteLoopbackApiBaseToPageHost(raw);
}

function dualPanels(dl: string, ul: string): PanelBoot[] {
  return [
    {
      id: "panel-dl-media",
      controlsRootId: "panel-dl-controls",
      apiBase: rewriteLoopbackApiBaseToPageHost(dl),
      playbackPath: "live_dl",
      modeSelectId: "mode-select-dl",
      heading: "Downlink",
    },
    {
      id: "panel-ul-media",
      controlsRootId: "panel-ul-controls",
      apiBase: rewriteLoopbackApiBaseToPageHost(ul),
      playbackPath: "live_ul",
      modeSelectId: "mode-select-ul",
      heading: "Uplink",
    },
  ];
}

function rxNaiPlaybackPaths(): { rx: string; nai: string } {
  const rx =
    import.meta.env.VITE_PLAYBACK_PATH_RX?.trim() || "live_rx";
  const nai =
    import.meta.env.VITE_PLAYBACK_PATH_NAI?.trim() || "live_nai";
  return { rx, nai };
}

function rxNaiPanels(rx: string, nai: string): PanelBoot[] {
  const paths = rxNaiPlaybackPaths();
  return [
    {
      id: "panel-ul-media",
      controlsRootId: "panel-ul-controls",
      apiBase: rewriteLoopbackApiBaseToPageHost(rx),
      playbackPath: paths.rx,
      modeSelectId: "mode-select-ul",
      heading: "RX",
    },
    {
      id: "panel-nai-media",
      controlsRootId: "panel-nai-controls",
      apiBase: rewriteLoopbackApiBaseToPageHost(nai),
      playbackPath: paths.nai,
      modeSelectId: "mode-select-nai",
      heading: "NAI",
    },
  ];
}

function combineLayout(): ResolvedGuiLayout {
  return {
    dual: false,
    layoutMode: "combine",
    panels: [
      {
        id: "panel-dl-media",
        controlsRootId: "panel-dl-controls",
        apiBase: singleApiBase(),
        playbackPath: "live",
        modeSelectId: "mode-select-dl",
        heading: "Live (MediaMTX)",
      },
    ],
  };
}

export function resolvePanels(): ResolvedGuiLayout {
  const mode = resolveGuiModeEnv();
  const dl = import.meta.env.VITE_API_BASE_DL?.trim();
  const ul = import.meta.env.VITE_API_BASE_UL?.trim();
  const rx =
    import.meta.env.VITE_API_BASE_RX?.trim() || ul;
  const nai = import.meta.env.VITE_API_BASE_NAI?.trim();

  if (mode === "combine") {
    return combineLayout();
  }

  if (mode === "dlul") {
    if (dl && ul) {
      return { dual: true, layoutMode: "dlul", panels: dualPanels(dl, ul) };
    }
    console.warn(
      "[resgrid] VITE_GUI_MODE=dlul but VITE_API_BASE_DL and/or VITE_API_BASE_UL is missing; showing DL/UL panels that have API bases (never NAI/combine).",
    );
    const panels: PanelBoot[] = [];
    if (dl) panels.push(dualPanels(dl, ul || dl)[0]);
    if (ul) panels.push(dualPanels(dl || ul, ul)[1]);
    return { dual: panels.length > 1, layoutMode: "dlul", panels };
  }

  if (mode === "rxnai") {
    if (rx && nai) {
      return { dual: true, layoutMode: "rxnai", panels: rxNaiPanels(rx, nai) };
    }
    console.warn(
      "[resgrid] VITE_GUI_MODE=rxnai but VITE_API_BASE_RX and/or VITE_API_BASE_NAI is missing; showing RX/NAI panels that have API bases (never DL/combine).",
    );
    const panels: PanelBoot[] = [];
    if (rx) panels.push(rxNaiPanels(rx, nai || rx)[0]);
    if (nai) panels.push(rxNaiPanels(rx || nai, nai)[1]);
    return { dual: panels.length > 1, layoutMode: "rxnai", panels };
  }

  // auto: infer from API bases (rx+nai checked before dl+ul so stray VITE_API_BASE_DL does not force DL/UL)
  if (rx && nai) {
    return { dual: true, layoutMode: "rxnai", panels: rxNaiPanels(rx, nai) };
  }
  if (dl && ul) {
    return { dual: true, layoutMode: "dlul", panels: dualPanels(dl, ul) };
  }

  return combineLayout();
}
