import { envTruthy } from "./env";

/**
 * When false, the UI will not call `hardReconnectPlayback` (publish-delay timers, WHEP onError,
 * HLS fatal loop, HLS stall heuristics in `ResgridPanel`). MediaMTX’s own WHEP 404 retry inside
 * `mediamtx-reader.js` still runs.
 *
 * Disable: `?playbackReconnect=0` or `VITE_PLAYBACK_AUTORECONNECT=0`. Enable explicitly: `=1`.
 */
export function playbackAutoreconnect(): boolean {
  if (typeof window !== "undefined") {
    const q = new URLSearchParams(window.location.search).get("playbackReconnect")?.trim().toLowerCase();
    if (q === "0" || q === "false" || q === "no" || q === "off") return false;
    if (q === "1" || q === "true" || q === "yes" || q === "on") return true;
  }
  const v = import.meta.env.VITE_PLAYBACK_AUTORECONNECT?.trim();
  if (v !== undefined && v !== "") {
    return envTruthy(v);
  }
  return true;
}
