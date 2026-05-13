import type { ResgridLogFn } from "../config/debug";

export type LocalPreviewRecorderButtons = {
  start: HTMLButtonElement;
  stop: HTMLButtonElement;
  play: HTMLButtonElement;
  backToLive: HTMLButtonElement;
  download: HTMLButtonElement;
};

function pickRecorderMime(): string {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

/**
 * Records the live &lt;video&gt; preview via {@link HTMLVideoElement.captureStream} + {@link MediaRecorder},
 * plays the result in an overlay element, and supports WebM download. Browser / CORS dependent.
 */
export class LocalPreviewRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private blob: Blob | null = null;
  private playbackUrl: string | null = null;
  private recording = false;
  private overlayVisible = false;

  private readonly onStartClick = (): void => void this.onStart();
  private readonly onStopClick = (): void => void this.onStop();
  private readonly onPlayClick = (): void => void this.onPlay();
  private readonly onLiveClick = (): void => void this.onBackToLive();
  private readonly onDownloadClick = (): void => this.onDownload();

  constructor(
    private readonly liveVideo: HTMLVideoElement,
    private readonly overlayVideo: HTMLVideoElement,
    private readonly buttons: LocalPreviewRecorderButtons,
    private readonly setPlaybackStatus: (t: string) => void,
    private readonly log: ResgridLogFn,
    /** Used in download filename, e.g. live_dl */
    private readonly fileStem: string,
  ) {
    this.buttons.start.addEventListener("click", this.onStartClick);
    this.buttons.stop.addEventListener("click", this.onStopClick);
    this.buttons.play.addEventListener("click", this.onPlayClick);
    this.buttons.backToLive.addEventListener("click", this.onLiveClick);
    this.buttons.download.addEventListener("click", this.onDownloadClick);
    this.syncButtons();
  }

  destroy(): void {
    this.buttons.start.removeEventListener("click", this.onStartClick);
    this.buttons.stop.removeEventListener("click", this.onStopClick);
    this.buttons.play.removeEventListener("click", this.onPlayClick);
    this.buttons.backToLive.removeEventListener("click", this.onLiveClick);
    this.buttons.download.removeEventListener("click", this.onDownloadClick);
    void this.teardownRecorder();
    this.onBackToLive();
    this.blob = null;
    this.chunks = [];
  }

  private syncButtons(): void {
    const hasBlob = Boolean(this.blob && this.blob.size > 0);
    this.buttons.start.disabled = this.recording || this.overlayVisible;
    this.buttons.stop.disabled = !this.recording;
    this.buttons.play.disabled = this.recording || !hasBlob || this.overlayVisible;
    this.buttons.backToLive.disabled = !this.overlayVisible;
    this.buttons.download.disabled = this.recording || !hasBlob;
  }

  private revokePlaybackUrl(): void {
    if (this.playbackUrl) {
      URL.revokeObjectURL(this.playbackUrl);
      this.playbackUrl = null;
    }
    this.overlayVideo.removeAttribute("src");
    this.overlayVideo.load();
  }

  private async teardownRecorder(): Promise<void> {
    const rec = this.mediaRecorder;
    if (!rec) return;
    if (rec.state === "recording" || rec.state === "paused") {
      await new Promise<void>((resolve) => {
        rec.addEventListener("stop", () => resolve(), { once: true });
        try {
          rec.requestData();
        } catch {
          /* optional */
        }
        try {
          rec.stop();
        } catch {
          resolve();
        }
      });
    }
    this.mediaRecorder = null;
    this.recording = false;
  }

  private async onStart(): Promise<void> {
    if (this.overlayVisible) void this.onBackToLive();
    await this.teardownRecorder();
    this.chunks = [];
    this.blob = null;

    const cap = (
      this.liveVideo as HTMLVideoElement & { captureStream?: (fps?: number) => MediaStream }
    ).captureStream;
    if (typeof cap !== "function") {
      this.setPlaybackStatus("Recording: captureStream() is not supported in this browser");
      this.log("local record", "no captureStream");
      return;
    }

    let stream: MediaStream;
    try {
      stream = cap.call(this.liveVideo);
    } catch (e) {
      this.setPlaybackStatus("Recording: could not capture the preview (CORS or not playing?)");
      this.log("local record captureStream error", e);
      return;
    }

    const tracks = stream.getVideoTracks();
    if (tracks.length === 0) {
      this.setPlaybackStatus("Recording: no video track — wait until the stream is playing, then try again");
      this.log("local record", "empty capture stream");
      return;
    }

    const mime = pickRecorderMime();
    try {
      this.mediaRecorder = mime
        ? new MediaRecorder(stream, { mimeType: mime })
        : new MediaRecorder(stream);
    } catch (e) {
      this.setPlaybackStatus("Recording: MediaRecorder failed to start");
      this.log("local record MediaRecorder ctor", e);
      return;
    }

    this.mediaRecorder.ondataavailable = (ev: BlobEvent) => {
      if (ev.data && ev.data.size > 0) this.chunks.push(ev.data);
    };

    this.mediaRecorder.onerror = (ev) => {
      this.log("local record MediaRecorder error", ev);
    };

    try {
      this.mediaRecorder.start(250);
    } catch (e) {
      this.mediaRecorder = null;
      this.setPlaybackStatus("Recording: start() failed");
      this.log("local record start", e);
      return;
    }

    this.recording = true;
    this.setPlaybackStatus("Recording… Stop when finished");
    this.syncButtons();
    this.log("local record started", this.mediaRecorder.mimeType);
  }

  private async onStop(): Promise<void> {
    if (!this.mediaRecorder || !this.recording) return;
    const rec = this.mediaRecorder;
    const blob = await new Promise<Blob | null>((resolve) => {
      rec.addEventListener(
        "stop",
        () => {
          const type = rec.mimeType || "video/webm";
          const b = new Blob(this.chunks, { type });
          resolve(b.size > 0 ? b : null);
        },
        { once: true },
      );
      try {
        rec.requestData();
      } catch {
        /* ignore */
      }
      try {
        rec.stop();
      } catch {
        resolve(null);
      }
    });
    this.mediaRecorder = null;
    this.recording = false;
    this.blob = blob;
    this.syncButtons();
    if (blob) {
      this.setPlaybackStatus(`Recording saved (${(blob.size / (1024 * 1024)).toFixed(2)} MiB) — Play or Download`);
      this.log("local record stopped", blob.size, "bytes", blob.type);
    } else {
      this.setPlaybackStatus("Recording stopped with no data — try again while the video is playing");
    }
  }

  private async onPlay(): Promise<void> {
    if (!this.blob || this.blob.size === 0 || this.recording) return;
    this.revokePlaybackUrl();
    this.playbackUrl = URL.createObjectURL(this.blob);
    this.overlayVideo.src = this.playbackUrl;
    this.overlayVideo.hidden = false;
    this.overlayVideo.removeAttribute("hidden");
    this.overlayVisible = true;
    this.syncButtons();
    try {
      await this.overlayVideo.play();
      this.setPlaybackStatus("Playing recorded clip — Back to live returns to the stream");
    } catch (e) {
      this.setPlaybackStatus("Could not play recording — try Download and open locally");
      this.log("local record play", e);
    }
  }

  private onBackToLive(): void {
    this.overlayVideo.pause();
    this.revokePlaybackUrl();
    this.overlayVideo.hidden = true;
    this.overlayVideo.setAttribute("hidden", "");
    this.overlayVisible = false;
    this.syncButtons();
    this.setPlaybackStatus("Live preview");
  }

  private onDownload(): void {
    if (!this.blob || this.blob.size === 0) return;
    const ext = this.blob.type.includes("mp4") ? "mp4" : "webm";
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(this.blob);
    a.download = `resgrid-${this.fileStem}-${stamp}.${ext}`;
    a.rel = "noopener";
    a.click();
    URL.revokeObjectURL(a.href);
    this.setPlaybackStatus("Download started");
    this.log("local record download", a.download);
  }
}
