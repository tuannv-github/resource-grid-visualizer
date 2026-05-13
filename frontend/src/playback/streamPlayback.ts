import "../mediamtx-reader.js";
import Hls from "hls.js";
import type { ResgridApi } from "../api/resgridApi";
import type { ResgridLogFn } from "../config/debug";
import { playbackMode } from "../config/playbackMode";
import type { PlaybackUrls } from "../domain/streamTypes";

export type StreamPlaybackDeps = {
  api: ResgridApi;
  video: HTMLVideoElement;
  playbackPath: string;
  setPlaybackStatus: (text: string) => void;
  logPlayback: ResgridLogFn;
  /** Forward verbose WHEP traces to console when true. */
  debugWhep: boolean;
  /** Used by HLS stall / video error reconnect paths. */
  getLastGenerating: () => boolean | undefined;
};

/**
 * Owns HLS + WHEP subscription lifecycle for one panel. Keeps DOM/video wiring out of control-plane UI.
 */
export class StreamPlayback {
  private readonly api: ResgridApi;
  private readonly video: HTMLVideoElement;
  private readonly playbackPath: string;
  private readonly setPlaybackStatus: (text: string) => void;
  private readonly logPlayback: ResgridLogFn;
  private readonly debugWhep: boolean;
  private readonly getLastGenerating: () => boolean | undefined;

  private hlsRef: Hls | null = null;
  private whepReader: MediaMTXWebRTCReader | null = null;
  private subscribedHlsUrl: string | null = null;
  private subscribedWhepUrl: string | null = null;
  private nativeHlsUrl: string | null = null;
  private nudgeTimerIds: number[] = [];
  private reconnectBusy = false;
  private playbackGraceUntilMs = 0;
  private lastTimeupdateAtMs = 0;
  private boundTimeupdate = (): void => {
    this.lastTimeupdateAtMs = Date.now();
  };
  private boundVideoError = (): void => {
    this.logPlayback("video element error", this.video.error);
    if (this.getLastGenerating() === true && playbackMode() === "hls") {
      void this.hardReconnectPlayback("video-element");
    }
  };

  constructor(deps: StreamPlaybackDeps) {
    this.api = deps.api;
    this.video = deps.video;
    this.playbackPath = deps.playbackPath;
    this.setPlaybackStatus = deps.setPlaybackStatus;
    this.logPlayback = deps.logPlayback;
    this.debugWhep = deps.debugWhep;
    this.getLastGenerating = deps.getLastGenerating;
  }

  getLastProgressAtMs(): number {
    return this.lastTimeupdateAtMs;
  }

  setGraceMs(until: number): void {
    this.playbackGraceUntilMs = until;
  }

  getGraceUntilMs(): number {
    return this.playbackGraceUntilMs;
  }

  attachVideoListeners(): void {
    this.video.addEventListener("timeupdate", this.boundTimeupdate);
    this.video.addEventListener("error", this.boundVideoError);
  }

  detachVideoListeners(): void {
    this.video.removeEventListener("timeupdate", this.boundTimeupdate);
    this.video.removeEventListener("error", this.boundVideoError);
  }

  private fallbackHlsUrl(): string | undefined {
    const env = import.meta.env.VITE_HLS_URL?.trim();
    if (env) return env;
    if (typeof window !== "undefined") {
      return `${window.location.protocol}//${window.location.hostname}:8888/${this.playbackPath}/index.m3u8`;
    }
    return undefined;
  }

  private async resolvePlaybackUrls(): Promise<PlaybackUrls | null> {
    const base = this.api.baseUrl;
    const apiUrl = `${base}/api/stream/playback`;
    try {
      const r = await fetch(apiUrl);
      this.logPlayback("GET", apiUrl, "→", r.status);
      if (r.ok) {
        const j = (await r.json()) as { hls?: string; webrtc_whep?: string };
        if (j.hls && j.webrtc_whep) {
          this.logPlayback("playback (from API)", j);
          return { hls: j.hls, webrtc_whep: j.webrtc_whep };
        }
      }
      this.logPlayback("playback API missing fields or non-OK, using fallbacks");
      this.setPlaybackStatus(`Playback API error ${r.status} — trying fallback URLs`);
    } catch (e) {
      this.logPlayback("GET", apiUrl, "failed", e);
      this.setPlaybackStatus(`Cannot reach ${base} — trying fallback playback URLs`);
    }

    const h = this.fallbackHlsUrl();
    if (!h) return null;
    const p = new URL(h);
    const scheme = p.protocol.replace(/:$/, "") || "http";
    const whep = `${scheme}://${p.hostname}:8889/${this.playbackPath}/whep`;
    const fb = { hls: h, webrtc_whep: whep };
    this.logPlayback("playback (fallback)", fb);
    return fb;
  }

  destroyWebRtc(): void {
    if (this.whepReader) {
      this.whepReader.close();
      this.whepReader = null;
    }
    this.subscribedWhepUrl = null;
    this.video.srcObject = null;
  }

  destroyHls(): void {
    if (this.hlsRef) {
      this.hlsRef.destroy();
      this.hlsRef = null;
    }
    this.subscribedHlsUrl = null;
    this.nativeHlsUrl = null;
    this.video.srcObject = null;
    this.video.removeAttribute("src");
  }

  /** Full teardown + re-subscribe (same effect as F5 for playback). */
  async hardReconnectPlayback(reason: string): Promise<void> {
    if (this.reconnectBusy) {
      this.logPlayback("hardReconnect skipped (busy)", reason);
      return;
    }
    this.reconnectBusy = true;
    try {
      this.logPlayback("hardReconnect", reason);
      this.setPlaybackStatus("Reconnecting stream…");
      this.lastTimeupdateAtMs = 0;
      this.destroyWebRtc();
      this.destroyHls();
      await this.ensurePlaybackSubscription();
    } finally {
      this.reconnectBusy = false;
    }
  }

  /** After the publisher starts, MediaMTX may need a moment before HLS/WHEP is ready. */
  nudgePlaybackAfterPublish(): void {
    for (const id of this.nudgeTimerIds) window.clearTimeout(id);
    this.nudgeTimerIds = [];
    this.logPlayback("nudgePlaybackAfterPublish");
    this.playbackGraceUntilMs = Date.now() + 22_000;

    if (playbackMode() === "hls") {
      this.setPlaybackStatus(
        `Publisher started — MediaMTX serves …/${this.playbackPath}/index.m3u8 only after RTSP ingest is up; a short 404 in the network tab is normal.`,
      );
      this.nudgeTimerIds.push(
        window.setTimeout(() => void this.hardReconnectPlayback("publish-hls-delayed"), 1200),
      );
      const delaysMs = [3200, 6200, 12_000];
      for (const d of delaysMs) {
        this.nudgeTimerIds.push(
          window.setTimeout(() => {
            this.logPlayback("nudgeHls after publish", d, "ms");
            this.nudgeHlsSubscription();
          }, d),
        );
      }
    } else {
      void this.hardReconnectPlayback("publish");
      this.nudgeTimerIds.push(
        window.setTimeout(() => void this.hardReconnectPlayback("whep-followup"), 450),
      );
      this.nudgeTimerIds.push(
        window.setTimeout(() => void this.hardReconnectPlayback("whep-followup2"), 1200),
      );
    }
  }

  clearNudgeTimers(): void {
    for (const id of this.nudgeTimerIds) window.clearTimeout(id);
    this.nudgeTimerIds = [];
  }

  /** Reload manifest/segments without dropping subscription (same MediaMTX URL). */
  nudgeHlsSubscription(): void {
    if (this.hlsRef) {
      this.hlsRef.startLoad();
      return;
    }
    if (this.nativeHlsUrl && this.video.src) {
      this.video.load();
    }
  }

  private ensureWebRtcSubscription(whepUrl: string): void {
    this.destroyHls();
    if (this.whepReader && this.subscribedWhepUrl === whepUrl) return;
    this.destroyWebRtc();
    this.subscribedWhepUrl = whepUrl;
    const Reader = window.MediaMTXWebRTCReader;
    if (!Reader) {
      this.setPlaybackStatus("MediaMTX WebRTC reader failed to load");
      return;
    }
    this.logPlayback("WHEP subscribe", whepUrl);
    this.whepReader = new Reader({
      url: whepUrl,
      debugLog: this.debugWhep
        ? (...args: unknown[]) => console.log("[resgrid/whep]", ...args)
        : undefined,
      onError: (err) => {
        this.logPlayback("WebRTC onError", err);
        this.setPlaybackStatus(`WebRTC: ${err}`);
        window.setTimeout(() => void this.hardReconnectPlayback("whep-error"), 600);
      },
      onTrack: (evt) => {
        this.logPlayback("WebRTC onTrack", evt.track.kind, evt.streams[0]?.id);
        this.video.srcObject = evt.streams[0];
        this.setPlaybackStatus(
          "WebRTC live — Start/Stop only affect generation on the backend",
        );
        void this.video.play().catch(() => {
          this.setPlaybackStatus("WebRTC — click the video if autoplay was blocked");
        });
      },
    });
  }

  private wireHls(hls: Hls): void {
    let fatalRecoveryAttempts = 0;

    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      fatalRecoveryAttempts = 0;
      this.setPlaybackStatus(
        "LL-HLS (short segments; still some buffer). Use default WebRTC for real-time.",
      );
      this.video.play().catch(() => {
        this.setPlaybackStatus("LL-HLS — click the video if the browser blocked autoplay");
      });
    });

    hls.on(Hls.Events.ERROR, (_, data) => {
      this.logPlayback("Hls.Events.ERROR", data.type, data.details, "fatal=", data.fatal);
      const resp = (data as { response?: { code?: number } }).response;
      if (
        data.details === Hls.ErrorDetails.MANIFEST_LOAD_ERROR &&
        resp?.code === 404 &&
        Date.now() < this.playbackGraceUntilMs
      ) {
        this.setPlaybackStatus(
          "Waiting for HLS manifest (404) — publisher may still be connecting to MediaMTX; hls.js will retry.",
        );
      }
      if (!data.fatal) return;
      fatalRecoveryAttempts += 1;
      if (fatalRecoveryAttempts >= 3) {
        fatalRecoveryAttempts = 0;
        void this.hardReconnectPlayback("hls-fatal");
        return;
      }
      this.setPlaybackStatus("HLS stall… (waiting for MediaMTX / generation)");
      switch (data.type) {
        case Hls.ErrorTypes.NETWORK_ERROR:
          window.setTimeout(() => hls.startLoad(), 400);
          break;
        case Hls.ErrorTypes.MEDIA_ERROR:
          hls.recoverMediaError();
          break;
        default:
          window.setTimeout(() => hls.startLoad(), 700);
          break;
      }
    });
  }

  /** Subscribe once; Start/Stop do not tear this down — only generation on the backend. */
  async ensurePlaybackSubscription(): Promise<void> {
    const urls = await this.resolvePlaybackUrls();
    if (!urls) {
      this.setPlaybackStatus(
        "No playback URL (backend unreachable and no VITE_HLS_URL / host:8888 fallback)",
      );
      return;
    }

    if (playbackMode() === "webrtc") {
      this.setPlaybackStatus(`Subscribing WebRTC (WHEP) — ${urls.webrtc_whep}`);
      this.ensureWebRtcSubscription(urls.webrtc_whep);
      return;
    }

    this.destroyWebRtc();
    const url = urls.hls;

    if (Hls.isSupported()) {
      if (this.hlsRef && this.subscribedHlsUrl === url) {
        this.nudgeHlsSubscription();
        return;
      }
      if (this.hlsRef) {
        this.hlsRef.destroy();
        this.hlsRef = null;
      }
      this.subscribedHlsUrl = url;
      this.nativeHlsUrl = null;
      this.setPlaybackStatus(`Subscribing LL-HLS — ${url}`);

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 0,
        maxBufferLength: 5,
        liveSyncDurationCount: 1,
        liveMaxLatencyDurationCount: 10,
        manifestLoadingTimeOut: 8000,
        manifestLoadingMaxRetry: 12,
        levelLoadingMaxRetry: 12,
        fragLoadingMaxRetry: 12,
        fragLoadingRetryDelay: 200,
        manifestLoadingRetryDelay: 200,
        levelLoadingRetryDelay: 200,
      });
      this.hlsRef = hls;
      hls.loadSource(url);
      hls.attachMedia(this.video);
      this.wireHls(hls);
      return;
    }

    if (this.video.canPlayType("application/vnd.apple.mpegurl")) {
      if (this.nativeHlsUrl !== url) {
        this.nativeHlsUrl = url;
        this.video.src = url;
        this.video.addEventListener(
          "loadedmetadata",
          () => {
            this.setPlaybackStatus("Subscribed (native HLS)");
            this.video.play().catch(() => {});
          },
          { once: true },
        );
      } else {
        this.nudgeHlsSubscription();
      }
      return;
    }

    this.setPlaybackStatus("HLS is not supported — try Chromium with WebRTC (?playback=webrtc)");
  }

  destroy(): void {
    this.clearNudgeTimers();
    this.detachVideoListeners();
    this.destroyWebRtc();
    this.destroyHls();
  }
}
