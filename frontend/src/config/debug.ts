import { envTruthy } from "./env";

/** Console diagnostics for API + MediaMTX + WebRTC. Enable with `?debug=1` or `VITE_DEBUG=1`. */
export function isPlaybackDebugEnabled(): boolean {
  if (typeof window === "undefined") return false;
  const q = new URLSearchParams(window.location.search).get("debug");
  if (q === "1" || q === "true" || q === "yes") return true;
  return envTruthy(import.meta.env.VITE_DEBUG);
}

export type ResgridLogFn = (...args: unknown[]) => void;

export function createResgridLogger(debug: boolean): ResgridLogFn {
  if (!debug) return () => {};
  return (...args: unknown[]) => {
    console.log("[resgrid]", ...args);
  };
}
