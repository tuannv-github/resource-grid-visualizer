import "../mediamtx-reader.js";
import Hls from "hls.js";
import type { ResgridApi } from "../api/resgridApi";
import type { ResgridLogFn } from "../config/debug";
import { envTruthy } from "../config/env";
import { playbackAutoreconnect } from "../config/playbackAutoreconnect";
import { playbackMode } from "../config/playbackMode";
import type { PlaybackUrls } from "../domain/publisherStatus";

export type MediaPlaybackDeps = {
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
export class MediaPlayback {
  private readonly api: ResgridApi;
  private readonly video: HTMLVideoElement;
  private readonly playbackPath: string;
  private readonly setPlaybackStatus: (text: string) => void;
  private readonly logPlayback: ResgridLogFn;
  private readonly debugWhep: boolean;
  private readonly getLastGenerating: () => boolean | undefined;

  private hlsRef: Hls | null = null;
  private whepReader: MediaMTXWebRTCReader | null = null;
  /** One inbound **MediaStream** for all WHEP tracks so a second `ontrack` (audio) cannot replace video with an audio-only **MediaStream**. */
  private webrtcInboundMediaStream: MediaStream | null = null;
  private subscribedHlsUrl: string | null = null;
  private subscribedWhepUrl: string | null = null;
  private nativeHlsUrl: string | null = null;
  private nudgeTimerIds: number[] = [];
  private reconnectBusy = false;
  private playbackGraceUntilMs = 0;
  private lastTimeupdateAtMs = 0;
  /** Coalesce `play()` across video+audio `ontrack` (stacked `play()` → AbortError + rVFC churn). */
  private webrtcPlayDebounceTimer: number | null = null;
  /** Start rVFC FPS monitor once per WHEP session; `startReceivedFrameFpsMonitor` stops any prior loop. */
  private webrtcFpsMonitorStarted = false;
  /** One-shot hint when video dimensions stay 0 after connect (ICE / MTX_WEBRTC_ADDITIONAL_HOSTS). */
  private webrtcDecodeHintTimer: number | null = null;
  /** Frames presented in the current 1s window (requestVideoFrameCallback) or last quality sample (fallback). */
  private receivedFpsFrames = 0;
  private receivedFpsRvfcHandle: number | null = null;
  private receivedFpsLogIntervalId: number | null = null;
  private receivedFpsFallbackLastTotal: number | null = null;
  private boundTimeupdate = (): void => {
    this.lastTimeupdateAtMs = Date.now();
  };
  private boundVideoError = (): void => {
    this.logPlayback("video element error", this.video.error);
    if (this.getLastGenerating() === true && playbackMode() === "hls") {
      void this.hardReconnectPlayback("video-element");
    }
  };
  private boundVideoWaiting = (): void => {
    if (!this.isFpsVerbose()) return;
    console.log("[resgrid/video]", this.playbackPath, "waiting", this.snapshotVideoForDebug());
  };
  private boundVideoStalled = (): void => {
    if (!this.isFpsVerbose()) return;
    console.log("[resgrid/video]", this.playbackPath, "stalled", this.snapshotVideoForDebug());
  };
  private boundVideoSuspend = (): void => {
    if (!this.isFpsVerbose()) return;
    console.log("[resgrid/video]", this.playbackPath, "suspend", this.snapshotVideoForDebug());
  };

  constructor(deps: MediaPlaybackDeps) {
    this.api = deps.api;
    this.video = deps.video;
    this.playbackPath = deps.playbackPath;
    this.setPlaybackStatus = deps.setPlaybackStatus;
    this.logPlayback = deps.logPlayback;
    this.debugWhep = deps.debugWhep;
    this.getLastGenerating = deps.getLastGenerating;
  }

  /** Verbose playback / FPS logging: `?debug=1` / `VITE_DEBUG` or `?fpsdebug=1` (extra lines without full WHEP SDP trace). */
  private isFpsVerbose(): boolean {
    if (this.debugWhep) return true;
    if (typeof window === "undefined") return false;
    const q = new URLSearchParams(window.location.search).get("fpsdebug");
    return q === "1" || q === "true" || q === "yes";
  }

  private snapshotVideoForDebug(): Record<string, unknown> {
    const v = this.video;
    const so = v.srcObject;
    const stream = so instanceof MediaStream ? so : null;
    return {
      paused: v.paused,
      muted: v.muted,
      readyState: v.readyState,
      networkState: v.networkState,
      videoWidth: v.videoWidth,
      videoHeight: v.videoHeight,
      currentTime: v.currentTime,
      src: v.currentSrc?.slice(0, 120) ?? "",
      hasSrcObject: Boolean(so),
      streamTracks:
        stream?.getTracks().map((t) => ({
          kind: t.kind,
          enabled: t.enabled,
          muted: t.muted,
          readyState: t.readyState,
        })) ?? [],
    };
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

  private stopReceivedFrameFpsMonitor(): void {
    if (this.receivedFpsRvfcHandle !== null) {
      try {
        this.video.cancelVideoFrameCallback(this.receivedFpsRvfcHandle);
      } catch {
        /* ignore */
      }
      this.receivedFpsRvfcHandle = null;
    }
    if (this.receivedFpsLogIntervalId !== null) {
      window.clearInterval(this.receivedFpsLogIntervalId);
      this.receivedFpsLogIntervalId = null;
    }
    this.receivedFpsFallbackLastTotal = null;
  }

  /** Logs approximate decoded/displayed frame rate to the console about once per second. */
  private startReceivedFrameFpsMonitor(): void {
    this.stopReceivedFrameFpsMonitor();
    this.receivedFpsFrames = 0;

    const logSecond = (fpsApprox: number, via: "rVFC" | "playbackQuality"): void => {
      const rounded = Math.round(fpsApprox * 10) / 10;
      console.log("[resgrid/fps]", this.playbackPath, "~", rounded, "fps", via);
      if (!this.isFpsVerbose()) return;
      const snap = this.snapshotVideoForDebug();
      console.log("[resgrid/fps-debug]", this.playbackPath, {
        via,
        fpsInLast1s: rounded,
        playbackMode: playbackMode(),
        whepUrl: this.subscribedWhepUrl,
        hlsUrl: this.subscribedHlsUrl,
        ...snap,
      });
      if (rounded < 1) {
        console.log(
          "[resgrid/fps-hint]",
          "0 fps: paused / not presenting, or rVFC not firing. Remote WebRTC tracks often show muted=true until RTP arrives (normal). videoWidth≈0 → decode/ICE/publisher.",
        );
      }
    };

    const v = this.video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    };
    if (typeof v.requestVideoFrameCallback === "function") {
      if (this.isFpsVerbose()) {
        console.log("[resgrid/fps]", this.playbackPath, "monitor start: requestVideoFrameCallback + 1s tick");
      }
      const onFrame = (): void => {
        this.receivedFpsFrames += 1;
        this.receivedFpsRvfcHandle = v.requestVideoFrameCallback!(onFrame);
      };
      this.receivedFpsRvfcHandle = v.requestVideoFrameCallback(onFrame);
      this.receivedFpsLogIntervalId = window.setInterval(() => {
        const n = this.receivedFpsFrames;
        this.receivedFpsFrames = 0;
        logSecond(n, "rVFC");
      }, 1000);
      return;
    }

    if (this.isFpsVerbose()) {
      console.log(
        "[resgrid/fps]",
        this.playbackPath,
        "monitor start: getVideoPlaybackQuality (no requestVideoFrameCallback in this browser)",
      );
    }
    this.receivedFpsFallbackLastTotal =
      this.video.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
    this.receivedFpsLogIntervalId = window.setInterval(() => {
      const q = this.video.getVideoPlaybackQuality?.();
      const total = q?.totalVideoFrames ?? this.receivedFpsFallbackLastTotal ?? 0;
      const prev = this.receivedFpsFallbackLastTotal ?? total;
      const delta = total - prev;
      this.receivedFpsFallbackLastTotal = total;
      logSecond(delta, "playbackQuality");
    }, 1000);
  }

  attachVideoListeners(): void {
    this.video.addEventListener("timeupdate", this.boundTimeupdate);
    this.video.addEventListener("error", this.boundVideoError);
    this.video.addEventListener("waiting", this.boundVideoWaiting);
    this.video.addEventListener("stalled", this.boundVideoStalled);
    this.video.addEventListener("suspend", this.boundVideoSuspend);
  }

  detachVideoListeners(): void {
    this.video.removeEventListener("timeupdate", this.boundTimeupdate);
    this.video.removeEventListener("error", this.boundVideoError);
    this.video.removeEventListener("waiting", this.boundVideoWaiting);
    this.video.removeEventListener("stalled", this.boundVideoStalled);
    this.video.removeEventListener("suspend", this.boundVideoSuspend);
  }

  private fallbackHlsUrl(): string | undefined {
    const env = import.meta.env.VITE_HLS_URL?.trim();
    if (env) return env;
    if (typeof window !== "undefined") {
      return `${window.location.protocol}//${window.location.hostname}:8888/${this.playbackPath}/index.m3u8`;
    }
    return undefined;
  }

  /** True when API/fallback URLs target this panel's MediaMTX path (e.g. live_rx, live_nai). */
  private urlsMatchPlaybackPath(urls: PlaybackUrls): boolean {
    const needle = `/${this.playbackPath}/`;
    return urls.hls.includes(needle) && urls.webrtc_whep.includes(needle);
  }

  private buildPlaybackUrlsForHost(hlsSample: string): PlaybackUrls {
    const p = new URL(hlsSample);
    const scheme = p.protocol.replace(/:$/, "") || "http";
    const hls = `${scheme}://${p.host}/${this.playbackPath}/index.m3u8`;
    const whep = `${scheme}://${p.hostname}:8889/${this.playbackPath}/whep`;
    return { hls, webrtc_whep: whep };
  }

  private buildFallbackPlaybackUrls(): PlaybackUrls | null {
    const h = this.fallbackHlsUrl();
    if (!h) return null;
    return this.buildPlaybackUrlsForHost(h);
  }

  /** Signaling URLs follow the page host so a baked LAN address is not required. */
  private alignPlaybackHostToPage(urls: PlaybackUrls): PlaybackUrls {
    if (typeof window === "undefined" || !window.location.hostname) return urls;
    const fix = (raw: string, port: string): string => {
      const u = new URL(raw);
      u.hostname = window.location.hostname;
      if (window.location.protocol === "http:" || window.location.protocol === "https:") {
        u.protocol = window.location.protocol;
      }
      u.port = port;
      return u.toString();
    };
    return {
      hls: fix(urls.hls, "8888"),
      webrtc_whep: fix(urls.webrtc_whep, "8889"),
    };
  }

  private async resolvePlaybackUrls(): Promise<PlaybackUrls | null> {
    const base = this.api.baseUrl;
    const apiUrl = `${base}/api/plotter/playback`;
    try {
      const r = await fetch(apiUrl);
      this.logPlayback("GET", apiUrl, "→", r.status);
      if (r.ok) {
        const j = (await r.json()) as { hls?: string; webrtc_whep?: string };
        if (j.hls && j.webrtc_whep) {
          const fromApi = { hls: j.hls, webrtc_whep: j.webrtc_whep };
          if (this.urlsMatchPlaybackPath(fromApi)) {
            const aligned = this.alignPlaybackHostToPage(fromApi);
            this.logPlayback("playback (from API)", aligned);
            return aligned;
          }
          this.logPlayback(
            "playback API path mismatch for",
            this.playbackPath,
            "— using MediaMTX fallback URLs",
            fromApi,
          );
          const corrected = this.alignPlaybackHostToPage(this.buildPlaybackUrlsForHost(fromApi.hls));
          this.logPlayback("playback (path-corrected from API host)", corrected);
          return corrected;
        }
      }
      this.logPlayback("playback API missing fields or non-OK, using fallbacks");
      this.setPlaybackStatus(`Playback API error ${r.status} — trying fallback URLs`);
    } catch (e) {
      this.logPlayback("GET", apiUrl, "failed", e);
      this.setPlaybackStatus(`Cannot reach ${base} — trying fallback playback URLs`);
    }

    const fb = this.buildFallbackPlaybackUrls();
    if (!fb) return null;
    const aligned = this.alignPlaybackHostToPage(fb);
    this.logPlayback("playback (fallback)", aligned);
    return aligned;
  }

  private releaseWebRtcInboundMediaStream(): void {
    if (this.webrtcInboundMediaStream) {
      for (const t of this.webrtcInboundMediaStream.getTracks()) {
        t.stop();
      }
      this.webrtcInboundMediaStream = null;
    }
    this.video.srcObject = null;
  }

  destroyWebRtc(): void {
    this.stopReceivedFrameFpsMonitor();
    if (this.webrtcPlayDebounceTimer !== null) {
      window.clearTimeout(this.webrtcPlayDebounceTimer);
      this.webrtcPlayDebounceTimer = null;
    }
    if (this.webrtcDecodeHintTimer !== null) {
      window.clearTimeout(this.webrtcDecodeHintTimer);
      this.webrtcDecodeHintTimer = null;
    }
    this.webrtcFpsMonitorStarted = false;
    if (this.whepReader) {
      try {
        this.whepReader.close?.();
      } catch (e) {
        this.logPlayback("destroyWebRtc close failed", e);
      }
      this.whepReader = null;
    }
    this.subscribedWhepUrl = null;
    this.releaseWebRtcInboundMediaStream();
  }

  destroyHls(): void {
    this.stopReceivedFrameFpsMonitor();
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
    if (!playbackAutoreconnect()) {
      this.logPlayback("hardReconnect skipped (playbackReconnect=0 / VITE_PLAYBACK_AUTORECONNECT=0)", reason);
      return;
    }
    if (this.reconnectBusy) {
      this.logPlayback("hardReconnect skipped (busy)", reason);
      return;
    }
    this.reconnectBusy = true;
    try {
      this.logPlayback("hardReconnect", reason);
        this.setPlaybackStatus("Reconnecting playback…");
      this.lastTimeupdateAtMs = 0;
      this.destroyWebRtc();
      this.destroyHls();
      await this.ensurePlaybackSubscription();
    } finally {
      this.reconnectBusy = false;
    }
  }

  /**
   * Publish-edge nudge timers used to call `hardReconnectPlayback` unconditionally; that tore down healthy
   * WHEP sessions (notably ~3.8s after `generating` flipped true). Skip when decode is already live.
   */
  private shouldSkipDelayedPublishReconnect(): boolean {
    if (this.video.error) return false;
    if (playbackMode() === "webrtc") {
      return Boolean(
        this.whepReader &&
          this.subscribedWhepUrl &&
          this.video.videoWidth > 0 &&
          this.video.videoHeight > 0,
      );
    }
    if (playbackMode() === "hls") {
      if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return false;
      return this.hlsRef !== null || Boolean(this.nativeHlsUrl);
    }
    return false;
  }

  private delayedPublishHardReconnect(reason: string): void {
    if (this.shouldSkipDelayedPublishReconnect()) {
      this.logPlayback("hardReconnect skipped (playback already healthy)", reason);
      return;
    }
    void this.hardReconnectPlayback(reason);
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
      if (playbackAutoreconnect()) {
        this.nudgeTimerIds.push(
          window.setTimeout(() => this.delayedPublishHardReconnect("publish-hls-delayed"), 1200),
        );
      }
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
      /* WHEP: MediaMTXWebRTCReader already backs off ~1.2s on 404 until RTSP ingest is up. Immediate + rapid
       * hardReconnects destroy that reader, reset backoff, and open many short-lived WHEP sessions (noisy logs). */
      this.setPlaybackStatus(
        `Publisher started — WebRTC may show 404 until RTSP is live on …/${this.playbackPath}; optional reconnect if playback is still not decoding.`,
      );
      if (playbackAutoreconnect()) {
        this.nudgeTimerIds.push(
          window.setTimeout(() => this.delayedPublishHardReconnect("publish-whep-delayed"), 1100),
        );
        this.nudgeTimerIds.push(
          window.setTimeout(() => this.delayedPublishHardReconnect("publish-whep-delayed2"), 3800),
        );
      }
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

  /** Debounced `play()` + single FPS monitor start after WHEP tracks settle (avoids AbortError + rVFC reset storm). */
  private scheduleWebRtcPlayAndMaybeStartFps(): void {
    if (this.webrtcPlayDebounceTimer !== null) {
      window.clearTimeout(this.webrtcPlayDebounceTimer);
    }
    this.webrtcPlayDebounceTimer = window.setTimeout(() => {
      this.webrtcPlayDebounceTimer = null;
      void this.video
        .play()
        .then(() => {
          if (this.isFpsVerbose()) {
            console.log(
              "[resgrid/webrtc-play]",
              this.playbackPath,
              "play() resolved",
              this.snapshotVideoForDebug(),
            );
          }
          if (!this.webrtcFpsMonitorStarted) {
            this.webrtcFpsMonitorStarted = true;
            this.startReceivedFrameFpsMonitor();
          }
          this.scheduleWebrtcNoDecodeHint();
          if (this.isFpsVerbose()) {
            window.setTimeout(() => {
              console.log(
                "[resgrid/webrtc-play+1.5s]",
                this.playbackPath,
                this.snapshotVideoForDebug(),
              );
            }, 1500);
          }
        })
        .catch((e) => {
          if (this.isFpsVerbose()) {
            console.warn("[resgrid/webrtc-play]", this.playbackPath, "play() rejected", e);
          }
          this.setPlaybackStatus("WebRTC — click the video if autoplay was blocked");
        });
    }, 48);
  }

  /** If still no decoded frames after a few seconds, surface the usual LAN/docker ICE fix (not only console). */
  private scheduleWebrtcNoDecodeHint(): void {
    if (this.webrtcDecodeHintTimer !== null) {
      window.clearTimeout(this.webrtcDecodeHintTimer);
    }
    this.webrtcDecodeHintTimer = window.setTimeout(() => {
      this.webrtcDecodeHintTimer = null;
      if (!this.whepReader || playbackMode() !== "webrtc") return;
      if (this.video.videoWidth > 0 || this.video.videoHeight > 0) return;
      let host = "";
      try {
        if (this.subscribedWhepUrl) host = new URL(this.subscribedWhepUrl).hostname;
      } catch {
        /* ignore */
      }
      const tip =
        `WebRTC: signaling OK but no decoded video yet. On LAN/docker set MediaMTX env MTX_WEBRTC_ADDITIONAL_HOSTS to ${host || "this host's LAN IP"} (same as the WHEP URL host), open firewall TCP+UDP 8189, redeploy. Or try ?playback=hls. If the publisher sends audio, rebuild with VITE_WHEP_INCLUDE_AUDIO=1.`;
      this.setPlaybackStatus(tip);
      console.warn("[resgrid/webrtc]", this.playbackPath, tip);
    }, 6500);
  }

  private ensureWebRtcSubscription(whepUrl: string): void {
    /* Must short-circuit before destroyHls(): destroyHls always clears video.srcObject and would
     * detach a healthy WebRTC MediaStream when ensurePlaybackSubscription runs again (e.g. status refresh). */
    if (this.whepReader && this.subscribedWhepUrl === whepUrl) return;
    this.destroyHls();
    this.destroyWebRtc();
    this.subscribedWhepUrl = whepUrl;
    const Reader = window.MediaMTXWebRTCReader;
    if (!Reader) {
      this.setPlaybackStatus("MediaMTX WebRTC reader failed to load");
      return;
    }
    this.logPlayback("WHEP subscribe", whepUrl);
    if (this.isFpsVerbose()) {
      console.log("[resgrid/webrtc]", this.playbackPath, "subscribe", whepUrl);
    }
    this.whepReader = new Reader({
      url: whepUrl,
      whepIncludeAudio: envTruthy(import.meta.env.VITE_WHEP_INCLUDE_AUDIO),
      debugLog: this.debugWhep
        ? (...args: unknown[]) => console.log("[resgrid/whep]", ...args)
        : undefined,
      onError: (err) => {
        this.logPlayback("WebRTC onError", err);
        const msg = String(err);
        /* MediaMTXWebRTCReader already backs off (retryPause) on 404 "no stream"; hardReconnect would
         * destroy it and spawn a new session ~600ms later → log spam and WHEP 404 storms. */
        if (msg.includes("retrying") || /stream not found|no stream/i.test(msg)) {
          this.setPlaybackStatus(
            "Waiting for WebRTC (WHEP 404 until a publisher is on this path) — retrying automatically; start plotter / generation if idle.",
          );
          return;
        }
        this.setPlaybackStatus(`WebRTC: ${err}`);
        window.setTimeout(() => void this.hardReconnectPlayback("whep-error"), 600);
      },
      onTrack: (evt) => {
        if (evt.track.kind === "audio" && !envTruthy(import.meta.env.VITE_WHEP_INCLUDE_AUDIO)) {
          evt.track.stop();
          this.logPlayback("WebRTC onTrack ignored (video-only WHEP)", evt.track.id);
          return;
        }
        if (this.webrtcInboundMediaStream === null) {
          this.webrtcInboundMediaStream = new MediaStream();
        }
        const ms = this.webrtcInboundMediaStream;
        if (!ms.getTracks().includes(evt.track)) {
          /* WHEP reconnect: new sender track IDs — drop prior same-kind tracks so we do not stack dead receivers. */
          for (const t of [...ms.getTracks()]) {
            if (t.kind === evt.track.kind && t.id !== evt.track.id) {
              ms.removeTrack(t);
            }
          }
          ms.addTrack(evt.track);
        }
        if (this.video.srcObject !== ms) {
          this.video.srcObject = ms;
        }
        this.logPlayback(
          "WebRTC onTrack",
          evt.track.kind,
          "streamTracks=",
          ms.getTracks().map((t) => t.kind).join(","),
        );
        if (this.isFpsVerbose()) {
          console.log("[resgrid/webrtc-track]", this.playbackPath, {
            kind: evt.track.kind,
            id: evt.track.id,
            label: evt.track.label,
            enabled: evt.track.enabled,
            muted: evt.track.muted,
            readyState: evt.track.readyState,
            streamId: evt.streams[0]?.id,
            mergedTrackKinds: ms.getTracks().map((t) => t.kind),
          });
        }
        this.setPlaybackStatus(
          "WebRTC live — Start/Stop only affect generation on the backend",
        );
        evt.track.addEventListener("unmute", () => this.scheduleWebRtcPlayAndMaybeStartFps());
        this.scheduleWebRtcPlayAndMaybeStartFps();
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
      this.startReceivedFrameFpsMonitor();
    });

    hls.on(Hls.Events.ERROR, (_: unknown, data: unknown) => {
      const d = data as {
        type: string;
        details: string;
        fatal: boolean;
        response?: { code?: number };
      };
      this.logPlayback("Hls.Events.ERROR", d.type, d.details, "fatal=", d.fatal);
      const resp = d.response;
      if (
        d.details === Hls.ErrorDetails.MANIFEST_LOAD_ERROR &&
        resp?.code === 404 &&
        Date.now() < this.playbackGraceUntilMs
      ) {
        this.setPlaybackStatus(
          "Waiting for HLS manifest (404) — publisher may still be connecting to MediaMTX; hls.js will retry.",
        );
      }
      if (!d.fatal) return;
      fatalRecoveryAttempts += 1;
      if (fatalRecoveryAttempts >= 3) {
        fatalRecoveryAttempts = 0;
        void this.hardReconnectPlayback("hls-fatal");
        return;
      }
      this.setPlaybackStatus("HLS stall… (waiting for MediaMTX / generation)");
      switch (d.type) {
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
    if (this.isFpsVerbose()) {
      console.log("[resgrid/playback-urls]", this.playbackPath, urls ?? "(null)");
    }
    if (!urls) {
      this.setPlaybackStatus(
        "No playback URL (backend unreachable and no VITE_HLS_URL / host:8888 fallback)",
      );
      return;
    }

    if (playbackMode() === "webrtc") {
      this.setPlaybackStatus(
        `Subscribing WebRTC (WHEP) — ${urls.webrtc_whep} (404 on POST is normal until RTSP ingest is live on this path)`,
      );
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
            void this.video
              .play()
              .then(() => {
                this.startReceivedFrameFpsMonitor();
              })
              .catch(() => {});
          },
          { once: true },
        );
      } else {
        this.nudgeHlsSubscription();
      }
      return;
    }

    this.setPlaybackStatus("HLS is not supported in this browser — try Chromium with WebRTC (WHEP).");
  }

  destroy(): void {
    this.clearNudgeTimers();
    this.detachVideoListeners();
    this.destroyWebRtc();
    this.destroyHls();
  }
}
