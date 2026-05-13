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
  /** When both DL and UL are set, the UI shows two columns (docker-compose.dlul.yaml). */
  readonly VITE_API_BASE_DL?: string;
  readonly VITE_API_BASE_UL?: string;
  /** Force single-column UI (one `ResgridPanel`) even if both API bases are set. Values: 1, true, yes, on. */
  readonly VITE_GUI_COMBINE?: string;
  /** Force dual DL/UL columns when both API bases exist (or set with dlul compose). Values: 1, true, yes, on. */
  readonly VITE_GUI_DLUL?: string;
  readonly VITE_HLS_URL?: string;
  readonly VITE_PLAYBACK_MODE?: string;
  /** Set to "1" to log connectivity / WebRTC diagnostics (or use ?debug=1). */
  readonly VITE_DEBUG?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
