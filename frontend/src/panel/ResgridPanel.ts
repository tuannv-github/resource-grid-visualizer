import { ResgridApi } from "../api/resgridApi";
import { createResgridLogger, isPlaybackDebugEnabled } from "../config/debug";
import { playbackMode } from "../config/playbackMode";
import {
  cudaPlotterRunningFromStatus,
  ffmpegPublisherRunningFromStatus,
  isIqCudaStreamMode,
  normalizeUiStreamMode,
  showsSyntheticIqPatternUi,
  STREAM_MODE_SYNTHETIC_VIDEO,
} from "../domain/streamModes";
import type { StreamStatus } from "../domain/streamTypes";
import { StreamPlayback } from "../playback/streamPlayback";
import { queryRole, requireRole } from "../ui/dom";
import { renderIqAxes } from "../ui/iqAxes";
import { renderIqChannelLegend, renderIqLegendError } from "../ui/iqLegend";
import { mountPublisherModeRadios } from "../ui/publisherModeRadios";

const debugEnabled = isPlaybackDebugEnabled();
const logPlayback = createResgridLogger(debugEnabled);

export type ResgridPanelOptions = {
  apiBase: string;
  /** MediaMTX path segment for HLS/WHEP fallback (live, live_dl, live_ul). */
  playbackPath: string;
  /** Distinct per panel so DL/UL radios do not interfere. */
  streamRadioName: string;
  /**
   * Status + configuration DOM (dashboard right column). When omitted, the same
   * element as the media root is used (single-root layout).
   */
  controlsRoot?: HTMLElement;
};

export class ResgridPanel {
  private readonly mediaRoot: HTMLElement;
  private readonly controlsRoot: HTMLElement;
  private readonly api: ResgridApi;
  readonly playbackPath: string;
  private readonly streamRadioName: string;
  private readonly playback: StreamPlayback;
  private iqPatternLabels = new Map<number, string>();
  private videoAxes: HTMLElement;
  private video: HTMLVideoElement;
  private statusEl: HTMLParagraphElement;
  private pubRunning: HTMLElement;
  private pubEncoder: HTMLElement;
  private pubSource: HTMLElement;
  private pubPath: HTMLElement;
  private pubIqProducerFps: HTMLElement;
  private pubPipelineStages: HTMLElement | null;
  private pubOutFps: HTMLElement;
  private pubFrames: HTMLElement;
  private pubMode: HTMLElement;
  private pubPlotter: HTMLElement;
  private pubGenerator: HTMLElement;
  private btnPlotterStart: HTMLButtonElement;
  private btnPlotterStop: HTMLButtonElement;
  private btnGeneratorStart: HTMLButtonElement;
  private btnGeneratorStop: HTMLButtonElement;
  private statusPoll: ReturnType<typeof setInterval> | null = null;
  private capturerPoll: ReturnType<typeof setInterval> | null = null;
  private lastGenerating: boolean | undefined = undefined;
  private controlsHydratedFromServer = false;

  constructor(mediaRoot: HTMLElement, opts: ResgridPanelOptions) {
    this.mediaRoot = mediaRoot;
    this.controlsRoot = opts.controlsRoot ?? mediaRoot;
    this.api = new ResgridApi(opts.apiBase);
    this.playbackPath = opts.playbackPath;
    this.streamRadioName = opts.streamRadioName;
    this.videoAxes = requireRole(this.mediaRoot, "video-axes");
    this.video = requireRole(this.mediaRoot, "video");
    this.statusEl = requireRole(this.controlsRoot, "status");
    this.pubRunning = requireRole(this.controlsRoot, "pub-running");
    this.pubEncoder = requireRole(this.controlsRoot, "pub-encoder");
    this.pubSource = requireRole(this.controlsRoot, "pub-source");
    this.pubPath = requireRole(this.controlsRoot, "pub-path");
    this.pubIqProducerFps = requireRole(this.controlsRoot, "pub-iq-producer-fps");
    this.pubPipelineStages = queryRole(this.controlsRoot, "pub-pipeline-stages");
    this.pubOutFps = requireRole(this.controlsRoot, "pub-out-fps");
    this.pubFrames = requireRole(this.controlsRoot, "pub-frames");
    this.pubMode = requireRole(this.controlsRoot, "pub-mode");
    this.pubPlotter = requireRole(this.controlsRoot, "pub-plotter");
    this.pubGenerator = requireRole(this.controlsRoot, "pub-generator");
    this.btnPlotterStart = requireRole(this.controlsRoot, "btn-plotter-start");
    this.btnPlotterStop = requireRole(this.controlsRoot, "btn-plotter-stop");
    this.btnGeneratorStart = requireRole(this.controlsRoot, "btn-generator-start");
    this.btnGeneratorStop = requireRole(this.controlsRoot, "btn-generator-stop");

    this.playback = new StreamPlayback({
      api: this.api,
      video: this.video,
      playbackPath: this.playbackPath,
      setPlaybackStatus: (t) => this.setPlaybackStatus(t),
      logPlayback,
      debugWhep: debugEnabled,
      getLastGenerating: () => this.lastGenerating,
    });
  }

  private setPlaybackStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  private initPublisherModeRadios(): void {
    const fs = queryRole<HTMLElement>(this.controlsRoot, "publisher-mode-fieldset");
    if (!fs) return;
    mountPublisherModeRadios(fs, this.streamRadioName);
  }

  async loadIqPatternCatalog(): Promise<void> {
    const j = await this.api.fetchIqPatternCatalog();
    if (!j) return;
    this.iqPatternLabels = new Map(j.patterns.map((p) => [p.id, p.label]));
    const sel = queryRole<HTMLSelectElement>(this.controlsRoot, "iq-pattern-select");
    if (!sel) return;
    const keep = sel.value;
    sel.replaceChildren();
    for (const p of j.patterns) {
      const o = document.createElement("option");
      o.value = String(p.id);
      o.textContent = p.label;
      sel.appendChild(o);
    }
    if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
  }

  async loadIqChannelLegend(): Promise<void> {
    const section = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
    const dlList = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-list-dl");
    const ulList = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-list-ul");
    const notes = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-notes");
    if (!section || !dlList || !ulList || !notes) return;

    const result = await this.api.getIqChannelLegend(logPlayback);
    if (result.ok) {
      renderIqChannelLegend(this.mediaRoot, result.payload);
    } else {
      renderIqLegendError(this.mediaRoot, result.message);
    }
  }

  private updateControlPanelIqVisibility(): void {
    const mode = this.selectedPublisherMode();
    const iq = isIqCudaStreamMode(mode);
    const cmdWrap = queryRole<HTMLElement>(this.controlsRoot, "iq-generator-fps-wrap");
    const patWrap = queryRole<HTMLElement>(this.controlsRoot, "iq-pattern-wrap");
    const genGroup = queryRole<HTMLElement>(this.controlsRoot, "generator-control-group");
    if (cmdWrap) cmdWrap.hidden = !iq;
    if (patWrap) patWrap.hidden = !showsSyntheticIqPatternUi(mode);
    if (genGroup) genGroup.hidden = !iq;
    const capGroup = queryRole<HTMLElement>(this.controlsRoot, "capturer-panel");
    if (capGroup) capGroup.hidden = !iq;
  }

  private hydrateControlsFromServerOnce(s: StreamStatus): void {
    if (this.controlsHydratedFromServer) return;
    this.controlsHydratedFromServer = true;
    const uiMode = normalizeUiStreamMode(s.stream_mode);
    const modeRadio = this.controlsRoot.querySelector<HTMLInputElement>(
      `input[name="${this.streamRadioName}"][value="${uiMode}"]`,
    );
    if (modeRadio) modeRadio.checked = true;
    const cmdInput = queryRole<HTMLInputElement>(this.controlsRoot, "iq-generator-fps");
    if (
      cmdInput &&
      s.iq_producer_command_fps != null &&
      !Number.isNaN(s.iq_producer_command_fps)
    ) {
      cmdInput.value = String(Math.round(s.iq_producer_command_fps * 10) / 10);
    }
    const patSel = queryRole<HTMLSelectElement>(this.controlsRoot, "iq-pattern-select");
    if (
      showsSyntheticIqPatternUi(uiMode) &&
      patSel &&
      s.iq_pattern != null &&
      !Number.isNaN(s.iq_pattern)
    ) {
      const v = String(Math.trunc(s.iq_pattern));
      if ([...patSel.options].some((o) => o.value === v)) patSel.value = v;
    }
    this.updateControlPanelIqVisibility();
  }

  selectedPublisherMode(): string {
    const el = this.controlsRoot.querySelector<HTMLInputElement>(
      `input[name="${this.streamRadioName}"]:checked`,
    );
    return el?.value ?? STREAM_MODE_SYNTHETIC_VIDEO;
  }

  private clearPipelineStages(): void {
    const el = this.pubPipelineStages;
    if (!el) return;
    el.replaceChildren();
    el.classList.remove("panel-status-pipeline--active");
    const ph = document.createElement("span");
    ph.className = "pipeline-stages-placeholder";
    ph.textContent = "—";
    el.appendChild(ph);
  }

  /**
   * Per-stage **output** throughput only (Hz): each value is what leaves that stage and is the **input**
   * to the next (Assembly → Raster → Encoder → RTSP).
   */
  private renderPipelineStagesPanel(s: StreamStatus): void {
    const el = this.pubPipelineStages;
    if (!el) return;
    const hzStr = (v: number | null | undefined): string =>
      v != null && !Number.isNaN(v) ? v.toFixed(1) : "—";
    const rout = s.raster_output_image_s ?? s.iq_plotter_fps;
    const encOut = s.encoder_output_fps ?? s.stats?.fps ?? null;
    el.classList.add("panel-status-pipeline--active");
    el.replaceChildren();

    const block = (heading: string, outHz: string): HTMLDivElement => {
      const wrap = document.createElement("div");
      wrap.className = "pipeline-block";
      const ht = document.createElement("div");
      ht.className = "pipeline-block__title";
      ht.textContent = heading;
      const row = document.createElement("div");
      row.className = "pipeline-row pipeline-row--single";
      const m = document.createElement("div");
      m.className = "pipeline-metric";
      const lb = document.createElement("span");
      lb.className = "pipeline-metric__label";
      lb.textContent = "Output";
      const vl = document.createElement("span");
      vl.className = "pipeline-metric__value";
      vl.textContent = outHz;
      m.append(lb, vl);
      row.append(m);
      wrap.append(ht, row);
      return wrap;
    };

    const root = document.createElement("div");
    root.className = "pipeline-stages";
    root.append(
      block("Assembly → Raster input", hzStr(s.assembly_output_slot_s)),
      block("Raster → Encoder input", hzStr(rout)),
      block("Encoder → RTSP / viewer", hzStr(encOut)),
    );
    el.appendChild(root);
  }

  applyStreamStatus(s: StreamStatus): void {
    this.hydrateControlsFromServerOnce(s);

    this.pubMode.textContent = s.stream_mode;
    this.pubRunning.textContent = s.generating ? "Yes" : "No";
    this.pubEncoder.textContent = s.video_encoder;
    this.pubSource.textContent = s.lavfi_video ?? "—";
    this.pubPath.textContent = s.mtx_path;
    const st = s.stats ?? null;
    const activeIq = isIqCudaStreamMode(s.stream_mode);
    this.pubOutFps.textContent =
      st?.fps != null && !Number.isNaN(st.fps) ? st.fps.toFixed(1) : "—";
    this.pubFrames.textContent = st?.frame != null ? String(st.frame) : "—";
    this.pubIqProducerFps.textContent =
      activeIq && s.iq_producer_fps != null && !Number.isNaN(s.iq_producer_fps)
        ? s.iq_producer_fps.toFixed(1)
        : "—";

    if (this.pubPipelineStages) {
      if (!activeIq || !s.generating) {
        this.clearPipelineStages();
      } else {
        this.renderPipelineStagesPanel(s);
      }
    }

    const cudaPlotterUp = cudaPlotterRunningFromStatus(s);
    const ffmpegUp = ffmpegPublisherRunningFromStatus(s);
    const plotterRunning = cudaPlotterUp || ffmpegUp;
    this.pubPlotter.textContent = plotterRunning ? "Running" : "Stopped";

    const genCapable = s.iq_generator_capable === true;
    const genRunning = s.iq_generator_running === true;
    if (!activeIq) this.pubGenerator.textContent = "—";
    else if (!s.generating) this.pubGenerator.textContent = "—";
    else if (!genCapable) this.pubGenerator.textContent = "N/A";
    else this.pubGenerator.textContent = genRunning ? "Running" : "Stopped";

    const capWrap = queryRole<HTMLElement>(this.controlsRoot, "capturer-progress-wrap");
    const capInflight = queryRole<HTMLProgressElement>(this.controlsRoot, "capturer-progress-inflight");
    const capWritten = queryRole<HTMLProgressElement>(this.controlsRoot, "capturer-progress-written");
    const capStat = queryRole<HTMLElement>(this.controlsRoot, "capturer-status");
    if (capInflight && capWritten && capStat) {
      if (!("capturer_capable" in s)) {
        if (capWrap) capWrap.hidden = true;
        capStat.textContent = "";
      } else {
        const capable = s.capturer_capable === true;
        const act = s.capturer_active === true;
        const tot = s.capturer_total ?? 0;
        const inflight = s.capturer_inflight ?? 0;
        const written = s.capturer_written ?? 0;
        const phase = s.capturer_phase ?? "idle";
        const dir = s.capturer_out_dir ?? "";
        const qDepth = s.capturer_queue_depth ?? 0;
        const qCap = s.capturer_queue_cap ?? 0;
        const jTh = s.capturer_jpeg_threads ?? 0;
        const diskQueue = (d: number, ifl: number) => `Disk ${d} / ${tot} · In-flight ${ifl}`;
        const queueDetail =
          act && tot > 0 && (qCap > 0 || jTh > 0)
            ? ` · Deque ${qDepth} / ${qCap}${jTh > 0 ? ` · ${jTh} JPEG thread${jTh === 1 ? "" : "s"}` : ""}`
            : "";
        const setBars = (show: boolean, ifl: number, d: number) => {
          if (capWrap) capWrap.hidden = !show;
          if (!show || tot <= 0) return;
          capInflight.max = tot;
          capWritten.max = tot;
          capInflight.value = Math.min(Math.max(0, ifl), tot);
          capWritten.value = Math.min(Math.max(0, d), tot);
        };
        if (!capable) {
          setBars(false, 0, 0);
          capStat.textContent =
            "Not configured: set RESGRID_CAPTURER_DIR (or legacy RESGRID_ENCODER_SNAP_DIR) and restart the plotter.";
        } else if (act && tot > 0) {
          if (phase === "capturing") {
            setBars(true, inflight, written);
            capStat.textContent = `${diskQueue(written, inflight)}${queueDetail}${dir ? ` → ${dir}` : ""}`;
          } else if (phase === "saving") {
            setBars(true, inflight, written);
            capStat.textContent = `${diskQueue(written, inflight)}${queueDetail}${dir ? ` → ${dir}` : ""}`;
          } else {
            setBars(true, Math.min(inflight, tot), Math.min(written, tot));
            capStat.textContent = `${diskQueue(written, inflight)}${queueDetail}${dir ? ` → ${dir}` : ""}`;
          }
        } else if (!act && tot > 0 && written >= tot) {
          setBars(true, tot, tot);
          capStat.textContent = `Done: ${diskQueue(written, inflight)}${dir ? ` → ${dir}` : ""}`;
        } else if (!act && tot > 0 && inflight === 0 && written === 0) {
          setBars(false, 0, 0);
          capStat.textContent =
            "Capturer did not finish (aborted or error); check plotter logs.";
        } else if (!act && tot > 0 && (inflight > 0 || written > 0) && written < tot) {
          setBars(true, inflight, written);
          capStat.textContent = `Interrupted: ${diskQueue(written, inflight)}${dir ? ` (${dir})` : ""}`;
        } else {
          setBars(false, 0, 0);
          capStat.textContent = "Ready.";
        }
        const capStart = queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-start");
        const capStop = queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-stop");
        if (capStart && capStop && "capturer_capable" in s) {
          const capable = s.capturer_capable === true;
          const act = s.capturer_active === true;
          const ph = s.capturer_phase ?? "idle";
          capStart.disabled = !capable || (act && (ph === "capturing" || ph === "saving"));
          capStop.disabled = !capable || !act || ph !== "capturing";
        }
      }
    }

    this.btnPlotterStart.disabled = cudaPlotterUp || ffmpegUp;
    this.btnPlotterStop.disabled = !(cudaPlotterUp || ffmpegUp);
    this.btnGeneratorStart.disabled =
      !activeIq || !s.generating || !genCapable || genRunning;
    this.btnGeneratorStop.disabled = !activeIq || !s.generating || !genRunning;

    const patRow = queryRole<HTMLElement>(this.controlsRoot, "iq-pattern-status-row");
    if (patRow) patRow.hidden = !showsSyntheticIqPatternUi(s.stream_mode);
    const pubPat = queryRole<HTMLElement>(this.controlsRoot, "pub-iq-pattern");
    if (pubPat) {
      if (
        !showsSyntheticIqPatternUi(s.stream_mode) ||
        s.iq_pattern == null ||
        Number.isNaN(s.iq_pattern)
      ) {
        pubPat.textContent = "—";
      } else {
        const id = Math.trunc(s.iq_pattern);
        pubPat.textContent = this.iqPatternLabels.get(id) ?? String(id);
      }
    }
    renderIqAxes(this.mediaRoot, s.iq_prb ?? 273, s.iq_symbols ?? 14, activeIq);

    const legendSec = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
    if (legendSec) legendSec.hidden = !activeIq;
    this.updateControlPanelIqVisibility();
  }

  async putIqPattern(): Promise<void> {
    if (!showsSyntheticIqPatternUi(this.selectedPublisherMode())) {
      this.setPlaybackStatus(
        "Synthetic IQ pattern applies only to SYNTHETIC_IQ (mode 2), not channel-based.",
      );
      return;
    }
    const sel = queryRole<HTMLSelectElement>(this.controlsRoot, "iq-pattern-select");
    if (!sel) return;
    const pattern = parseInt(sel.value, 10);
    if (!Number.isFinite(pattern)) {
      this.setPlaybackStatus("Pattern: invalid selection");
      return;
    }
    const url = `${this.api.baseUrl}/api/stream/iq/pattern`;
    try {
      const r = await this.api.putIqPattern(pattern, logPlayback);
      if (!r.ok) {
        const t = await r.text();
        this.setPlaybackStatus(`Pattern update failed: ${r.status} ${t.slice(0, 120)}`);
        return;
      }
      this.setPlaybackStatus("IQ pattern updated");
      await this.refreshPublisherStatus();
    } catch (e) {
      logPlayback("PUT", url, "failed", e);
      this.setPlaybackStatus("Pattern: network error");
    }
  }

  async putIqGeneratorFps(): Promise<void> {
    const input = queryRole<HTMLInputElement>(this.controlsRoot, "iq-generator-fps");
    if (!input) return;
    const fps = parseFloat(input.value);
    if (!Number.isFinite(fps)) {
      this.setPlaybackStatus("Generator FPS: invalid number");
      return;
    }
    const url = `${this.api.baseUrl}/api/stream/iq/generator-fps`;
    try {
      const r = await this.api.putIqGeneratorFps(fps, logPlayback);
      if (!r.ok) {
        const t = await r.text();
        this.setPlaybackStatus(
          `Generator FPS update failed: ${r.status} ${t.slice(0, 120)}`,
        );
        return;
      }
      this.setPlaybackStatus("Generator FPS updated");
      await this.refreshPublisherStatus();
    } catch (e) {
      logPlayback("PUT", url, "failed", e);
      this.setPlaybackStatus("Generator FPS: network error");
    }
  }

  async fetchStreamStatus(): Promise<StreamStatus | null> {
    return this.api.fetchStreamStatus(logPlayback);
  }

  async refreshPublisherStatus(): Promise<void> {
    const s = await this.fetchStreamStatus();
    if (s) {
      const prev = this.lastGenerating;
      this.lastGenerating = s.generating;
      if (s.generating && prev === false) {
        this.playback.setGraceMs(Date.now() + 14_000);
        this.playback.nudgePlaybackAfterPublish();
      }
      this.applyStreamStatus(s);

      if (
        s.generating &&
        playbackMode() === "hls" &&
        Date.now() > this.playback.getGraceUntilMs()
      ) {
        if (this.video.error) {
          void this.playback.hardReconnectPlayback("video.error");
        } else if (
          this.playback.getLastProgressAtMs() > 0 &&
          Date.now() - this.playback.getLastProgressAtMs() > 7500 &&
          !this.video.paused &&
          !this.video.seeking
        ) {
          void this.playback.hardReconnectPlayback("no-progress");
        }
      }
    } else {
      this.pubRunning.textContent = "Unknown (API unreachable)";
      this.pubMode.textContent = "—";
      this.pubEncoder.textContent = "—";
      this.pubSource.textContent = "—";
      this.pubPath.textContent = "—";
      this.pubOutFps.textContent = "—";
      this.pubFrames.textContent = "—";
      this.pubIqProducerFps.textContent = "—";
      const pubPatUnk = queryRole<HTMLElement>(this.controlsRoot, "pub-iq-pattern");
      const patRowUnk = queryRole<HTMLElement>(this.controlsRoot, "iq-pattern-status-row");
      if (pubPatUnk) pubPatUnk.textContent = "—";
      if (patRowUnk)
        patRowUnk.hidden = !showsSyntheticIqPatternUi(this.selectedPublisherMode());
      this.btnPlotterStart.disabled = false;
      this.btnPlotterStop.disabled = true;
      this.btnGeneratorStart.disabled = true;
      this.btnGeneratorStop.disabled = true;
      this.pubPlotter.textContent = "—";
      this.pubGenerator.textContent = "—";
      this.videoAxes.dataset.iqVisible = "false";
      const legendUnk = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
      if (legendUnk) legendUnk.hidden = true;
      this.clearPipelineStages();
      this.updateControlPanelIqVisibility();
    }
  }

  private async postStream(path: "start" | "stop"): Promise<{ status: string }> {
    const mode = this.selectedPublisherMode();
    const r = await this.api.postStream(path, mode, logPlayback);
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    return (await r.json()) as { status: string };
  }

  async onPlotterStart(): Promise<void> {
    const mode = this.selectedPublisherMode();
    this.btnPlotterStart.disabled = true;
    try {
      let j: { status: string };
      if (isIqCudaStreamMode(mode)) {
        const r = await this.api.postPlotterStart(mode, logPlayback);
        if (!r.ok) {
          const t = await r.text();
          throw new Error(t || r.statusText);
        }
        j = (await r.json()) as { status: string };
      } else {
        j = await this.postStream("start");
      }
      if (j.status === "started") {
        this.lastGenerating = false;
      }
      this.setPlaybackStatus(
        "Plotter started — connecting… If you use LL-HLS, the first manifest requests may 404 until RTSP publish is live.",
      );
      await this.refreshPublisherStatus();
    } catch (e) {
      this.setPlaybackStatus(
        `Plotter start failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      await this.refreshPublisherStatus();
    }
  }

  async onCapturerStart(): Promise<void> {
    const inp = queryRole<HTMLInputElement>(this.controlsRoot, "capturer-frames");
    const btn = queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-start");
    if (!inp || !btn) return;
    const frames = parseInt(inp.value, 10);
    if (!Number.isFinite(frames) || frames < 1) {
      this.setPlaybackStatus("Capturer: invalid frame count");
      return;
    }
    btn.disabled = true;
    let succeeded = false;
    try {
      const r = await this.api.postPlotterCapturer(frames, logPlayback);
      if (!r.ok) {
        const t = await r.text();
        this.setPlaybackStatus(`Capturer failed: ${r.status} ${t.slice(0, 160)}`);
        return;
      }
      succeeded = true;
      this.setPlaybackStatus("Capturer armed (parallel to video encoder)");
      this.startCapturerPoll();
    } catch {
      this.setPlaybackStatus("Capturer: network error");
    } finally {
      if (!succeeded) btn.disabled = false;
      else void this.refreshPublisherStatus();
    }
  }

  async onCapturerStop(): Promise<void> {
    const btn = queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-stop");
    if (btn) btn.disabled = true;
    try {
      const r = await this.api.postPlotterCapturerStop(logPlayback);
      if (!r.ok) {
        const t = await r.text();
        this.setPlaybackStatus(`Capturer stop failed: ${r.status} ${t.slice(0, 160)}`);
        return;
      }
      this.setPlaybackStatus("Capturer stop requested; finishing queued frames…");
    } catch {
      this.setPlaybackStatus("Capturer stop: network error");
    } finally {
      void this.refreshPublisherStatus();
    }
  }

  private startCapturerPoll(): void {
    if (this.capturerPoll) clearInterval(this.capturerPoll);
    this.capturerPoll = setInterval(() => {
      void (async () => {
        const st = await this.fetchStreamStatus();
        if (!st) return;
        this.applyStreamStatus(st);
        if (!st.capturer_active) {
          if (this.capturerPoll) clearInterval(this.capturerPoll);
          this.capturerPoll = null;
        }
      })();
    }, 400);
  }

  async onPlotterStop(): Promise<void> {
    const mode = this.selectedPublisherMode();
    this.btnPlotterStop.disabled = true;
    try {
      if (isIqCudaStreamMode(mode)) {
        const r = await this.api.postPlotterStop(logPlayback);
        if (!r.ok) {
          const t = await r.text();
          throw new Error(t || r.statusText);
        }
      } else {
        await this.postStream("stop");
      }
      await this.refreshPublisherStatus();
      this.setPlaybackStatus(
        "Plotter stopped — playback stays subscribed; WebRTC may idle until you start again.",
      );
    } catch (e) {
      this.setPlaybackStatus(
        `Plotter stop failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      await this.refreshPublisherStatus();
    }
  }

  private async postGenerator(path: "start" | "stop"): Promise<{ status: string; pid?: number }> {
    const r = await this.api.postGenerator(path, logPlayback);
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    return (await r.json()) as { status: string; pid?: number };
  }

  async onGeneratorStart(): Promise<void> {
    this.btnGeneratorStart.disabled = true;
    try {
      const j = await this.postGenerator("start");
      this.setPlaybackStatus(
        j.status === "already_running"
          ? "Generator already running"
          : "Generator started",
      );
      await this.refreshPublisherStatus();
    } catch (e) {
      this.setPlaybackStatus(
        `Generator start failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      await this.refreshPublisherStatus();
    }
  }

  async onGeneratorStop(): Promise<void> {
    this.btnGeneratorStop.disabled = true;
    try {
      const j = await this.postGenerator("stop");
      this.setPlaybackStatus(
        j.status === "not_running" ? "Generator was not running" : "Generator stopped",
      );
      await this.refreshPublisherStatus();
    } catch (e) {
      this.setPlaybackStatus(
        `Generator stop failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      await this.refreshPublisherStatus();
    }
  }

  private setSwaggerLink(): void {
    const swagger = queryRole<HTMLAnchorElement>(this.controlsRoot, "swagger-link");
    if (swagger) swagger.href = this.api.swaggerDocsHref();
  }

  start(): void {
    this.initPublisherModeRadios();
    for (const el of this.controlsRoot.querySelectorAll<HTMLInputElement>(
      `input[name="${this.streamRadioName}"]`,
    )) {
      el.addEventListener("change", () => this.updateControlPanelIqVisibility());
    }
    this.setSwaggerLink();
    this.btnPlotterStart.addEventListener("click", () => void this.onPlotterStart());
    this.btnPlotterStop.addEventListener("click", () => void this.onPlotterStop());
    this.btnGeneratorStart.addEventListener("click", () => void this.onGeneratorStart());
    this.btnGeneratorStop.addEventListener("click", () => void this.onGeneratorStop());
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-iq-generator-fps-apply")?.addEventListener(
      "click",
      () => void this.putIqGeneratorFps(),
    );
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-iq-pattern-apply")?.addEventListener("click", () =>
      void this.putIqPattern(),
    );
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-start")?.addEventListener("click", () =>
      void this.onCapturerStart(),
    );
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-stop")?.addEventListener("click", () =>
      void this.onCapturerStop(),
    );

    this.playback.attachVideoListeners();

    this.statusPoll = setInterval(() => void this.refreshPublisherStatus(), 2000);

    logPlayback("boot", {
      href: typeof window !== "undefined" ? window.location.href : "",
      apiBase: this.api.baseUrl,
      playback: playbackMode(),
      debug: debugEnabled,
    });
    void (async () => {
      await Promise.all([this.loadIqPatternCatalog(), this.loadIqChannelLegend()]);
      this.updateControlPanelIqVisibility();
      await this.refreshPublisherStatus();
      await this.playback.ensurePlaybackSubscription();
    })();
  }

  destroy(): void {
    if (this.statusPoll) clearInterval(this.statusPoll);
    this.statusPoll = null;
    if (this.capturerPoll) clearInterval(this.capturerPoll);
    this.capturerPoll = null;
    this.playback.destroy();
  }
}
