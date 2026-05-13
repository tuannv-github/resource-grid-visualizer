import { envTruthy } from "./env";

export type PanelBoot = {
  /** DOM root for video + IQ legend (left column). */
  id: string;
  /** DOM root for status, configuration, and actions (right column, inside `<details>`). */
  controlsRootId: string;
  apiBase: string;
  playbackPath: string;
  streamRadioName: string;
  heading: string;
};

export type GuiLayoutMode = "combine" | "dlul" | "auto";

export type ResolvedGuiLayout = {
  dual: boolean;
  panels: PanelBoot[];
  layoutMode: GuiLayoutMode;
};

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
      streamRadioName: "stream-mode-dl",
      heading: "Downlink",
    },
    {
      id: "panel-ul-media",
      controlsRootId: "panel-ul-controls",
      apiBase: rewriteLoopbackApiBaseToPageHost(ul),
      playbackPath: "live_ul",
      streamRadioName: "stream-mode-ul",
      heading: "Uplink",
    },
  ];
}

export function resolvePanels(): ResolvedGuiLayout {
  const wantCombine = envTruthy(import.meta.env.VITE_GUI_COMBINE);
  const wantDlul = envTruthy(import.meta.env.VITE_GUI_DLUL);
  const dl = import.meta.env.VITE_API_BASE_DL?.trim();
  const ul = import.meta.env.VITE_API_BASE_UL?.trim();

  if (wantCombine && wantDlul) {
    console.warn(
      "[resgrid] VITE_GUI_COMBINE and VITE_GUI_DLUL are both set; using combine (single column).",
    );
  }

  if (wantCombine) {
    return {
      dual: false,
      layoutMode: "combine",
      panels: [
        {
          id: "panel-dl-media",
          controlsRootId: "panel-dl-controls",
          apiBase: singleApiBase(),
          playbackPath: "live",
          streamRadioName: "stream-mode-dl",
          heading: "Live (MediaMTX)",
        },
      ],
    };
  }

  const autoDual = Boolean(dl && ul);
  if (wantDlul || autoDual) {
    if (dl && ul) {
      return {
        dual: true,
        layoutMode: wantDlul ? "dlul" : "auto",
        panels: dualPanels(dl, ul),
      };
    }
    if (wantDlul) {
      console.warn(
        "[resgrid] VITE_GUI_DLUL is set but VITE_API_BASE_DL and/or VITE_API_BASE_UL is missing; falling back to combine.",
      );
    }
  }

  return {
    dual: false,
    layoutMode: "auto",
    panels: [
      {
        id: "panel-dl-media",
        controlsRootId: "panel-dl-controls",
        apiBase: singleApiBase(),
        playbackPath: "live",
        streamRadioName: "stream-mode-dl",
        heading: "Live (MediaMTX)",
      },
    ],
  };
}
