/** `webrtc` = MediaMTX WHEP (nearest to real-time). `hls` = segment buffering (LL-HLS tuned). */
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
