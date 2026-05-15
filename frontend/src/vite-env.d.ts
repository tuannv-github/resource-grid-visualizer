/// <reference types="vite/client" />

declare class MediaMTXWebRTCReader {
  constructor(conf: {
    url: string;
    user?: string;
    pass?: string;
    token?: string;
    onError?: (err: string) => void;
    onTrack?: (evt: RTCTrackEvent) => void;
    onDataChannel?: (evt: RTCDataChannelEvent) => void;
    /** When true, negotiate recvonly audio as well (default off: video-only WHEP). */
    whepIncludeAudio?: boolean;
    /** From main.ts when ?debug=1 or VITE_DEBUG=1 */
    debugLog?: (...args: unknown[]) => void;
  });
  close(): void;
}

interface Window {
  MediaMTXWebRTCReader: typeof MediaMTXWebRTCReader;
}

interface ImportMetaEnv {
  /** When "1", single-panel API base is same-origin (empty); use with Vite `server.proxy` to the FastAPI container. */
  readonly VITE_API_RELATIVE?: string;
  /** Dev only: Vite proxy target for `/api`, `/docs`, `/openapi.json`, `/redoc` (e.g. `http://backend:8081` in Docker Compose). */
  readonly VITE_DEV_PROXY_TARGET?: string;
  readonly VITE_API_BASE?: string;
  /** When both DL and UL are set, the UI shows two columns (compose/docker-compose.dlul.yaml). */
  readonly VITE_API_BASE_DL?: string;
  readonly VITE_API_BASE_UL?: string;
  /** Second API for NAI column (compose/docker-compose.ulnai*.yaml). */
  readonly VITE_API_BASE_NAI?: string;
  /**
   * GUI layout (build-time). **`combine`** | **`dlul`** | **`ulnai`** | **`auto`** (default).
   * Compose: `VITE_GUI_MODE=dlul` for DL+UL stacks, `VITE_GUI_MODE=ulnai` for UL+NAI.
   */
  readonly VITE_GUI_MODE?: string;
  /** @deprecated Use `VITE_GUI_MODE=combine`. */
  readonly VITE_GUI_COMBINE?: string;
  /** @deprecated Use `VITE_GUI_MODE=dlul`. */
  readonly VITE_GUI_DLUL?: string;
  /** @deprecated Use `VITE_GUI_MODE=ulnai`. */
  readonly VITE_GUI_ULNAI?: string;
  readonly VITE_HLS_URL?: string;
  readonly VITE_PLAYBACK_MODE?: string;
  /** When "0", skip `hardReconnectPlayback` (see `playbackAutoreconnect.ts`). */
  readonly VITE_PLAYBACK_AUTORECONNECT?: string;
  /** When "1", WHEP offer includes recvonly audio (try if video-only never decodes with your publisher). */
  readonly VITE_WHEP_INCLUDE_AUDIO?: string;
  /** Set to "1" to log connectivity / WebRTC diagnostics (or use ?debug=1). */
  readonly VITE_DEBUG?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
