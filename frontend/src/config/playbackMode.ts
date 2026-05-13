/** Default: WebRTC (WHEP) unless `VITE_PLAYBACK_MODE=hls` or `?playback=hls`. */
export function playbackMode(): "webrtc" | "hls" {
  const q =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("playback")
      : null;
  if (q === "hls" || q === "webrtc") return q;
  const env = import.meta.env.VITE_PLAYBACK_MODE?.trim().toLowerCase();
  if (env === "hls") return "hls";
  return "webrtc";
}
