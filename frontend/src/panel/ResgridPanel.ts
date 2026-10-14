import { ResgridApi } from "../api/resgridApi";
import { createResgridLogger, isPlaybackDebugEnabled } from "../config/debug";
import { playbackMode } from "../config/playbackMode";
import {
  cudaHeatmapStatusRelevant,
  cudaPlotterRunningFromStatus,
  syntheticImagesFeedRunningFromStatus,
  isIqCudaMode,
  normalizeUiMode,
  SYNTHETIC_FAPI,
  SYNTHETIC_IMAGES,
  SYNTHETIC_RESOURCE_GRID,
} from "../domain/publisherModes";
import { panelVisibilityForMode } from "../domain/panelModeVisibility";
import { LAVFI_PRESETS_FOR_SYNTHETIC_IMAGES } from "../domain/syntheticImagesLavfiPresets";
import type { GeneratorStartBody } from "../domain/generatorModes";
import type { PublisherStatus } from "../domain/publisherStatus";
import { MediaPlayback } from "../playback/mediaPlayback";
import { queryRole, requireRole } from "../ui/dom";
import { renderIqAxes } from "../ui/iqAxes";
import { renderIqChannelLegend, renderIqLegendError, renderMagnitudeScale } from "../ui/iqLegend";
import { mountModeSelect } from "../ui/modeSelect";

/** One **SyntheticMqWriterStatus** slice from **GET /api/generator/status**. */
type FifoWriterSlice = {
  running: boolean;
  capable: boolean;
  pid?: number | null;
};

function formatFifoWriterStatus(slice: FifoWriterSlice | undefined, whenHidden: string): string {
  if (!slice) return whenHidden;
  if (slice.capable !== true) return "N/A";
  if (slice.running === true) {
    const p = slice.pid;
    return p != null && Number.isFinite(p) ? `Running (pid ${Math.trunc(p)})` : "Running";
  }
  return "Stopped";
}

const debugEnabled = isPlaybackDebugEnabled();
const logPlayback = createResgridLogger(debugEnabled);

/** Poll merged **GET /api/plotter/status** + **GET /api/generator/status** this often (via **`fetchPublisherStatus`**). */
const PUBLISHER_STATUS_POLL_MS = 1000;

export type ResgridPanelOptions = {
  apiBase: string;
  /** MediaMTX path segment for HLS/WHEP (live, live_dl, live_ul, live_rx, live_nai, …). */
  playbackPath: string;
  /** Distinct HTML **id** for this panel’s mode `<select>`. */
  modeSelectId: string;
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
  private readonly modeSelectId: string;
  private readonly playback: MediaPlayback;
  private videoAxes: HTMLElement;
  private video: HTMLVideoElement;
  private statusEl: HTMLParagraphElement;
  private pubPath: HTMLElement;
  private pubPublishing: HTMLElement;
  private pubIqProducerFps: HTMLElement;
  private pubPipelineAsm: HTMLElement | null;
  private pubPipelineRast: HTMLElement | null;
  private pubPipelineEnc: HTMLElement | null;
  private pubFrames: HTMLElement;
  private pubCudaPlotterStatus: HTMLElement;
  private pubGrid: HTMLElement | null;
  private pubIqSource: HTMLElement | null;
  private mediaStatus: HTMLElement | null;
  private mediaStatusPlotter: HTMLElement | null;
  private mediaStatusGrid: HTMLElement | null;
  private mediaStatusIq: HTMLElement | null;
  private mediaStatusPlay: HTMLElement | null;
  private mediaStatusHeld: HTMLElement | null = null;
  private mediaStatusInput: HTMLElement | null = null;
  private pubInputFps: HTMLElement | null = null;
  private metricInput: HTMLElement | null = null;
  private metricEncoder: HTMLElement | null = null;
  private metricGrid: HTMLElement | null = null;
  private metricState: HTMLElement | null = null;
  private metricPlotter: HTMLElement | null = null;
  private metricIq: HTMLElement | null = null;
  private metricPlayback: HTMLElement | null = null;
  private inputHeld = false;
  private inputRateLow = false;
  private playbackMediaState = "warn";
  private pubGenFapi: HTMLElement;
  private btnPlotterStart: HTMLButtonElement;
  private btnPlotterStop: HTMLButtonElement;
  private plotStartBtn: HTMLButtonElement | null = null;
  private plotStopBtn: HTMLButtonElement | null = null;
  private btnGeneratorStart: HTMLButtonElement;
  private btnGeneratorStop: HTMLButtonElement;
  private statusPoll: ReturnType<typeof setInterval> | null = null;
  private capturerPoll: ReturnType<typeof setInterval> | null = null;
  private lastGenerating: boolean | undefined = undefined;
  private controlsHydratedFromServer = false;
  private lastPublisherStatus: PublisherStatus | null = null;

  constructor(mediaRoot: HTMLElement, opts: ResgridPanelOptions) {
    this.mediaRoot = mediaRoot;
    this.controlsRoot = opts.controlsRoot ?? mediaRoot;
    this.api = new ResgridApi(opts.apiBase);
    this.playbackPath = opts.playbackPath;
    this.modeSelectId = opts.modeSelectId;
    this.videoAxes = requireRole(this.mediaRoot, "video-axes");
    this.video = requireRole(this.mediaRoot, "video");
    this.statusEl = requireRole(this.controlsRoot, "status");
    this.pubPath = requireRole(this.controlsRoot, "pub-path");
    this.pubPublishing = requireRole(this.controlsRoot, "pub-publishing");
    this.pubIqProducerFps = requireRole(this.controlsRoot, "pub-iq-producer-fps");
    this.pubPipelineAsm = queryRole(this.controlsRoot, "pub-pipeline-asm");
    this.pubPipelineRast = queryRole(this.controlsRoot, "pub-pipeline-rast");
    this.pubPipelineEnc = queryRole(this.controlsRoot, "pub-pipeline-enc");
    this.pubFrames = requireRole(this.controlsRoot, "pub-frames");
    this.pubCudaPlotterStatus = requireRole(this.controlsRoot, "pub-cuda-plotter-status");
    this.pubGrid = queryRole(this.controlsRoot, "pub-grid");
    this.pubIqSource = queryRole(this.controlsRoot, "pub-iq-source");
    this.mediaStatus = queryRole(this.mediaRoot, "media-status");
    this.mediaStatusPlotter = queryRole(this.mediaRoot, "media-status-plotter");
    this.mediaStatusGrid = queryRole(this.mediaRoot, "media-status-grid");
    this.mediaStatusIq = queryRole(this.mediaRoot, "media-status-iq");
    this.mediaStatusPlay = queryRole(this.mediaRoot, "media-status-play");
    this.mediaStatusHeld = this.ensureHeldWarning();
    this.mediaStatusInput = this.ensureInputFps();
    this.pubInputFps = queryRole(this.controlsRoot, "pub-input-fps");
    this.metricInput = queryRole(this.mediaRoot, "metric-input");
    this.metricEncoder = queryRole(this.mediaRoot, "metric-encoder");
    this.metricGrid = queryRole(this.mediaRoot, "metric-grid");
    this.metricState = queryRole(this.mediaRoot, "metric-state");
    this.metricPlotter = queryRole(this.mediaRoot, "metric-plotter");
    this.metricIq = queryRole(this.mediaRoot, "metric-iq");
    this.metricPlayback = queryRole(this.mediaRoot, "metric-playback");
    this.pubGenFapi = requireRole(this.controlsRoot, "pub-gen-fapi");
    this.btnPlotterStart = requireRole(this.controlsRoot, "btn-plotter-start");
    this.btnPlotterStop = requireRole(this.controlsRoot, "btn-plotter-stop");
    this.plotStartBtn = queryRole<HTMLButtonElement>(this.mediaRoot, "plot-start");
    this.plotStopBtn = queryRole<HTMLButtonElement>(this.mediaRoot, "plot-stop");
    this.btnGeneratorStart = requireRole(this.controlsRoot, "btn-generator-start");
    this.btnGeneratorStop = requireRole(this.controlsRoot, "btn-generator-stop");

    this.playback = new MediaPlayback({
      api: this.api,
      video: this.video,
      playbackPath: this.playbackPath,
      setPlaybackStatus: (t) => this.setPlaybackStatus(t),
      logPlayback,
      debugWhep: debugEnabled,
      getLastGenerating: () => this.lastGenerating,
    });
  }

  /** Below this the picture repeats the latest slot and the plot line warns. */
  private static readonly SLOW_INPUT_HZ = 24;

  private ensureInputFps(): HTMLElement | null {
    if (!this.mediaStatus) return null;
    const existing = queryRole<HTMLElement>(this.mediaStatus, "media-status-input");
    if (existing) return existing;
    const el = document.createElement("span");
    el.dataset.role = "media-status-input";
    el.textContent = "Input —";
    this.mediaStatus.insertBefore(el, this.mediaStatus.firstChild);
    return el;
  }

  private ensureHeldWarning(): HTMLElement | null {
    if (!this.mediaStatus) return null;
    const existing = queryRole<HTMLElement>(this.mediaStatus, "media-status-held");
    if (existing) return existing;
    const el = document.createElement("span");
    el.dataset.role = "media-status-held";
    el.hidden = true;
    this.mediaStatus.appendChild(el);
    return el;
  }

  private renderGridTable(s: PublisherStatus, inputLabel: string, inputWarn: boolean): void {
    const enc = s.encoder_output_fps ?? s.stats?.fps ?? null;
    const encKnown = enc != null && !Number.isNaN(enc);
    const gridText =
      s.iq_prb != null && s.iq_symbols != null ? `${s.iq_prb} PRB × ${s.iq_symbols} symbols` : "—";
    let state = "—";
    if (s.generating) {
      if (inputWarn) state = "Low input";
      else state = "Live";
    } else if (s.cuda_plotter_active) {
      state = "Plotter idle";
    } else {
      state = "Stopped";
    }
    const setCell = (el: HTMLElement | null, text: string, warn: boolean) => {
      if (!el) return;
      el.textContent = text;
      if (warn) el.dataset.state = "warn";
      else delete el.dataset.state;
    };
    const plotterText = s.cuda_plotter_active ? (s.generating ? "Publishing" : "Running") : "Stopped";
    const iqText = s.iq_endpoint?.trim() || "—";
    setCell(this.metricInput, s.generating ? inputLabel : "—", inputWarn && !!s.generating);
    setCell(this.metricEncoder, s.generating && encKnown ? `${enc.toFixed(1)} /s` : "—", false);
    setCell(this.metricGrid, gridText, false);
    setCell(this.metricPlotter, plotterText, !s.cuda_plotter_active);
    setCell(this.metricIq, iqText, false);
    if (this.metricIq) this.metricIq.title = iqText === "—" ? "" : iqText;
    setCell(this.metricState, state, inputWarn && !!s.generating);
  }

  private paintMediaStatus(): void {
    if (this.mediaStatus)
      this.mediaStatus.dataset.state =
        this.inputHeld || this.inputRateLow ? "warn" : this.playbackMediaState;
  }

  private shortPlayback(text: string): string {
    const lower = text.toLowerCase();
    if (/whep 404|waiting for webrtc|retrying automatically/.test(lower)) return "Waiting for stream";
    if (/click the video/.test(lower)) return "Click video to play";
    if (text.length > 42) return `${text.slice(0, 39)}…`;
    return text;
  }

  private setPlaybackStatus(text: string): void {
    this.statusEl.textContent = text;
    if (this.mediaStatusPlay) this.mediaStatusPlay.textContent = text;
    if (this.metricPlayback) {
      this.metricPlayback.textContent = this.shortPlayback(text);
      this.metricPlayback.title = text;
    }
    const lower = text.toLowerCase();
    let state = "warn";
    if (/error|fail|cannot|not supported|blocked|invalid|network error|no stream/.test(lower)) state = "bad";
    else if (
      /live|subscribed|playing|started|updated|saved|preview|stopped|download started|was not running/.test(
        lower,
      )
    ) {
      state = "ok";
    }
    this.playbackMediaState = state;
    this.statusEl.dataset.state = state;
    if (this.metricPlayback) {
      if (state === "bad" || state === "warn") this.metricPlayback.dataset.state = "warn";
      else delete this.metricPlayback.dataset.state;
    }
    this.paintMediaStatus();
  }

  private initModeSelect(): void {
    const fs = queryRole<HTMLElement>(this.controlsRoot, "mode-fieldset");
    if (!fs) return;
    mountModeSelect(fs, this.modeSelectId);
  }

  populateLavfiSourceSelect(): void {
    const sel = queryRole<HTMLSelectElement>(this.controlsRoot, "synthetic-images-lavfi-source");
    if (!sel) return;
    const keep = sel.value;
    sel.replaceChildren();
    const o0 = document.createElement("option");
    o0.value = "";
    o0.textContent = "Server default (LAVFI_VIDEO / built-in)";
    sel.appendChild(o0);
    for (const p of LAVFI_PRESETS_FOR_SYNTHETIC_IMAGES) {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = p.label;
      if (p.description) o.title = p.description;
      sel.appendChild(o);
    }
    if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
  }

  private buildGeneratorStartBody(uiMode: string): GeneratorStartBody {
    const n = normalizeUiMode(uiMode);
    if (n === SYNTHETIC_IMAGES) {
      const sel = queryRole<HTMLSelectElement>(this.controlsRoot, "synthetic-images-lavfi-source");
      const v = sel?.value.trim() ?? "";
      if (v === "") {
        return { feed: "images" };
      }
      return { feed: "images", images: { preset: v } };
    }
    if (n === SYNTHETIC_RESOURCE_GRID) {
      return { feed: "resource_grid" };
    }
    return { feed: "fapi" };
  }

  async loadIqChannelLegend(): Promise<void> {
    const section = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
    const dlList = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-list-dl");
    const ulList = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-list-ul");
    const notes = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-notes");

    const result = await this.api.getIqChannelLegend(logPlayback);
    if (result.ok) {
      renderMagnitudeScale(this.mediaRoot, result.payload.magnitude);
      if (section && dlList && ulList && notes) {
        renderIqChannelLegend(this.mediaRoot, result.payload);
      }
    } else {
      renderMagnitudeScale(this.mediaRoot, undefined);
      if (section && dlList && ulList && notes) {
        renderIqLegendError(this.mediaRoot, result.message);
      }
    }
  }

  private setStatusDlRowChildrenVisible(role: string, show: boolean): void {
    const g = queryRole<HTMLElement>(this.controlsRoot, role);
    if (!g) return;
    for (const el of g.querySelectorAll("dt, dd")) {
      (el as HTMLElement).hidden = !show;
    }
  }

  /** Applies {@link panelVisibilityForMode} to status rows, configuration rows, and control toolbars. */
  private updatePanelVisibilityForMode(): void {
    const mode = normalizeUiMode(this.selectedMode());
    const v = panelVisibilityForMode(mode);

    const cmdWrap = queryRole<HTMLElement>(this.controlsRoot, "generator-resource-grid-fps-wrap");
    const lavfiWrap = queryRole<HTMLElement>(this.controlsRoot, "synthetic-images-lavfi-wrap");
    const plotterGroup = queryRole<HTMLElement>(this.controlsRoot, "plotter-control-group");
    const genToolbar = queryRole<HTMLElement>(this.controlsRoot, "generator-toolbar");
    const capGroup = queryRole<HTMLElement>(this.controlsRoot, "capturer-panel");

    if (cmdWrap) cmdWrap.hidden = !v.showConfigGeneratorFps;
    if (lavfiWrap) lavfiWrap.hidden = mode !== SYNTHETIC_IMAGES;
    if (plotterGroup) plotterGroup.hidden = !v.showControlPlotter;
    if (genToolbar) genToolbar.hidden = false;
    if (capGroup) capGroup.hidden = !v.showControlPlotter;

    this.setStatusDlRowChildrenVisible("status-row-iq-feed-pub", v.showStatusIqFeedGeneratorRow);

    const fpsRow = queryRole<HTMLElement>(this.controlsRoot, "status-row-generator-fps");
    if (fpsRow) fpsRow.hidden = !v.showStatusIqDetailBlock;
  }

  private hydrateControlsFromServerOnce(s: PublisherStatus): void {
    if (this.controlsHydratedFromServer) return;
    this.controlsHydratedFromServer = true;
    const uiMode = normalizeUiMode(s.mode);
    const modeSel = queryRole<HTMLSelectElement>(this.controlsRoot, "mode-select");
    if (modeSel && [...modeSel.options].some((o) => o.value === uiMode)) modeSel.value = uiMode;
    const cmdInput = queryRole<HTMLInputElement>(this.controlsRoot, "generator-resource-grid-fps");
    if (
      cmdInput &&
      s.iq_producer_command_fps != null &&
      !Number.isNaN(s.iq_producer_command_fps)
    ) {
      cmdInput.value = String(Math.round(s.iq_producer_command_fps * 10) / 10);
    }
    this.updatePanelVisibilityForMode();
  }

  selectedMode(): string {
    const el = queryRole<HTMLSelectElement>(this.controlsRoot, "mode-select");
    return el?.value ?? SYNTHETIC_FAPI;
  }

  private clearPipelineStages(): void {
    if (this.pubPipelineAsm) this.pubPipelineAsm.textContent = "—";
    if (this.pubPipelineRast) this.pubPipelineRast.textContent = "—";
    if (this.pubPipelineEnc) this.pubPipelineEnc.textContent = "—";
  }

  /**
   * Per-stage **output** throughput (Hz): value leaving each stage toward the next
   * (Assembler → Raster → Encoder → RTSP).
   */
  private renderPipelineStagesPanel(s: PublisherStatus): void {
    const hzStr = (v: number | null | undefined): string =>
      v != null && !Number.isNaN(v) ? v.toFixed(1) : "—";
    const rout = s.raster_output_image_s ?? s.iq_plotter_fps;
    const encOut = s.encoder_output_fps ?? s.stats?.fps ?? null;
    if (this.pubPipelineAsm) this.pubPipelineAsm.textContent = hzStr(s.assembly_output_slot_s);
    if (this.pubPipelineRast) this.pubPipelineRast.textContent = hzStr(rout);
    if (this.pubPipelineEnc) this.pubPipelineEnc.textContent = hzStr(encOut);
  }

  applyPublisherStatus(s: PublisherStatus): void {
    this.hydrateControlsFromServerOnce(s);

    const cudaPlotterUp = cudaPlotterRunningFromStatus(s);
    const syntheticImagesFeedUp = syntheticImagesFeedRunningFromStatus(s);
    const heatmapRelevant = cudaHeatmapStatusRelevant(s);
    const uiMode = normalizeUiMode(this.selectedMode());
    const uiIqGenerator = isIqCudaMode(uiMode);
    const serverProcessIq = isIqCudaMode(s.mode);

    this.pubPath.textContent = s.mtx_path;
    this.pubPublishing.textContent = s.generating ? "Yes" : "No";
    this.pubPublishing.dataset.state = s.generating ? "ok" : "warn";
    this.pubCudaPlotterStatus.textContent = cudaPlotterUp ? "Running" : "Stopped";
    this.pubCudaPlotterStatus.dataset.state = cudaPlotterUp ? "ok" : "warn";
    const gridLabel = `${s.iq_prb ?? "—"} PRB × ${s.iq_symbols ?? "—"} symbols${
      s.iq_from_header ? " (from header)" : s.iq_status_detail ? ` (${s.iq_status_detail})` : ""
    }`;
    const iqLabel = s.iq_endpoint?.trim() || "—";
    if (this.pubGrid) this.pubGrid.textContent = gridLabel;
    if (this.pubIqSource) this.pubIqSource.textContent = iqLabel;
    if (this.mediaStatusPlotter) {
      this.mediaStatusPlotter.textContent = cudaPlotterUp
        ? s.generating
          ? "Plotter running · publishing"
          : "Plotter running · not publishing"
        : "Plotter stopped";
    }
    if (this.mediaStatusGrid) this.mediaStatusGrid.textContent = gridLabel;
    if (this.mediaStatusIq) this.mediaStatusIq.textContent = iqLabel;
    this.inputHeld = s.input_held === true && !!s.generating;
    const inputFps = s.assembly_input_msg_s;
    const inputKnown = inputFps != null && !Number.isNaN(inputFps);
    const inputSlow =
      !!s.generating && inputKnown && inputFps < ResgridPanel.SLOW_INPUT_HZ;
    this.inputRateLow = inputSlow;
    const inputWarn = inputSlow || (!inputKnown && this.inputHeld);
    const inputLabel = inputKnown ? `${inputFps.toFixed(1)} /s` : "—";
    let inputText = `Input ${inputLabel}`;
    if (inputWarn) inputText += " — low input rate";
    if (this.mediaStatusInput) {
      this.mediaStatusInput.textContent = s.generating ? inputText : "Input —";
      if (inputWarn && s.generating) this.mediaStatusInput.dataset.state = "warn";
      else delete this.mediaStatusInput.dataset.state;
    }
    if (this.pubInputFps) {
      this.pubInputFps.textContent = s.generating && inputKnown ? inputLabel : "—";
      if (inputWarn && s.generating) this.pubInputFps.dataset.state = "warn";
      else delete this.pubInputFps.dataset.state;
    }
    if (this.mediaStatusHeld) {
      const showHeld = !!s.generating && inputWarn;
      this.mediaStatusHeld.hidden = !showHeld;
      this.mediaStatusHeld.textContent = showHeld ? "Low input rate — showing latest slot" : "";
    }
    this.renderGridTable(s, inputKnown ? inputLabel : "—", inputWarn);
    this.paintMediaStatus();
    const st = s.stats ?? null;
    this.pubFrames.textContent = st?.frame != null ? String(st.frame) : "—";
    this.pubIqProducerFps.textContent =
      heatmapRelevant && s.iq_producer_fps != null && !Number.isNaN(s.iq_producer_fps)
        ? s.iq_producer_fps.toFixed(1)
        : "—";

    if (this.pubPipelineAsm || this.pubPipelineRast || this.pubPipelineEnc) {
      if (!heatmapRelevant || !s.generating) {
        this.clearPipelineStages();
      } else {
        this.renderPipelineStagesPanel(s);
      }
    }

    const srg = s.synthetic_resource_grid;
    const sfa = s.synthetic_fapi;
    /** True if any generator path is running (same for every UI **mode** so Stop stays clickable when switching the dropdown). */
    const anyGenRunning =
      syntheticImagesFeedUp || srg?.running === true || sfa?.running === true;
    const generatorModeKnown =
      uiMode === SYNTHETIC_IMAGES ||
      uiMode === SYNTHETIC_RESOURCE_GRID ||
      uiMode === SYNTHETIC_FAPI;

    const pubModeFifo = queryRole<HTMLElement>(this.controlsRoot, "pub-mode-fifo-path");
    if (pubModeFifo) {
      if (uiMode === SYNTHETIC_IMAGES) {
        const p = s.video_generator_mq_path?.trim();
        pubModeFifo.textContent = p || "—";
      } else if (uiMode === SYNTHETIC_RESOURCE_GRID) {
        const p = srg?.mq_path?.trim();
        pubModeFifo.textContent = p || "—";
      } else if (uiMode === SYNTHETIC_FAPI) {
        const p = sfa?.mq_path?.trim();
        pubModeFifo.textContent = p || "—";
      } else {
        pubModeFifo.textContent = "—";
      }
    }

    const iqLike = uiIqGenerator;
    const dash = "—";
    if (!iqLike) {
      this.pubGenFapi.textContent = dash;
    } else {
      this.pubGenFapi.textContent = formatFifoWriterStatus(sfa, dash);
    }

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
          capStat.textContent = "";
        } else if (act && tot > 0) {
          if (phase === "capturing" || phase === "saving") {
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

    /* Plotter accepts FIFO/env configuration; do not gate Start on UI **mode**. */
    this.syncPlotterButtons(cudaPlotterUp);
    const srcSel = queryRole<HTMLSelectElement>(this.controlsRoot, "synthetic-images-lavfi-source");
    if (srcSel) {
      srcSel.disabled = syntheticImagesFeedUp;
      const sid = s.video_generator_source;
      if (sid && [...srcSel.options].some((o) => o.value === sid)) {
        srcSel.value = sid;
      }
    }
    /* Stop follows **any** running generator; Start only when none are (server enforces a single feed). */
    this.btnGeneratorStart.disabled = !generatorModeKnown || anyGenRunning;
    this.btnGeneratorStop.disabled = !generatorModeKnown || !anyGenRunning;

    const totalSym = s.iq_symbols ?? 14;
    renderIqAxes(
      this.mediaRoot,
      s.iq_prb ?? 273,
      totalSym,
      heatmapRelevant,
      s.iq_symbol_start ?? 0,
      s.iq_symbol_stop ?? Math.max(0, totalSym - 1),
    );

    const legendSec = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
    if (legendSec) legendSec.hidden = !(uiIqGenerator || serverProcessIq);
    this.updatePanelVisibilityForMode();
    this.lastPublisherStatus = s;
  }

  async putIqGeneratorFps(): Promise<void> {
    const input = queryRole<HTMLInputElement>(this.controlsRoot, "generator-resource-grid-fps");
    if (!input) return;
    const fps = parseFloat(input.value);
    if (!Number.isFinite(fps)) {
      this.setPlaybackStatus("Generator FPS: invalid number");
      return;
    }
    const url = `${this.api.baseUrl}/api/generator/fps`;
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

  async fetchPublisherStatus(): Promise<PublisherStatus | null> {
    return this.api.fetchPublisherStatus(logPlayback);
  }

  async refreshPublisherStatus(): Promise<void> {
    const s = await this.fetchPublisherStatus();
    if (s) {
      const prev = this.lastGenerating;
      this.lastGenerating = s.generating;
      if (s.generating && prev === false) {
        this.playback.setGraceMs(Date.now() + 14_000);
        this.playback.nudgePlaybackAfterPublish();
      }
      this.applyPublisherStatus(s);

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
      this.pubPublishing.textContent = "—";
      this.pubPublishing.removeAttribute("data-state");
      this.pubPath.textContent = "—";
      this.pubFrames.textContent = "—";
      this.pubIqProducerFps.textContent = "—";
      this.syncPlotterButtons(false);
      this.btnGeneratorStart.disabled = true;
      this.btnGeneratorStop.disabled = true;
      this.pubGenFapi.textContent = "—";
      this.pubCudaPlotterStatus.textContent = "—";
      this.pubCudaPlotterStatus.removeAttribute("data-state");
      this.videoAxes.dataset.iqVisible = "false";
      const legendUnk = queryRole<HTMLElement>(this.mediaRoot, "iq-legend-section");
      if (legendUnk) legendUnk.hidden = true;
      this.clearPipelineStages();
      this.updatePanelVisibilityForMode();
    }
  }

  private plotterEncoderFpsStartOption(): number | undefined {
    const inp = queryRole<HTMLInputElement>(this.controlsRoot, "plotter-encoder-fps");
    if (!inp) return undefined;
    const raw = inp.value.trim();
    if (raw === "") return undefined;
    const n = Number(raw);
    if (!Number.isFinite(n)) return undefined;
    return Math.round(n);
  }

  private syncPlotterEncoderFpsInputFromResponse(j: {
    encoder_fps?: number | null;
  }): void {
    if (j.encoder_fps == null || Number.isNaN(j.encoder_fps)) return;
    const inp = queryRole<HTMLInputElement>(this.controlsRoot, "plotter-encoder-fps");
    if (!inp) return;
    inp.value = String(Math.trunc(j.encoder_fps));
  }

  private syncPlotterButtons(running: boolean): void {
    this.btnPlotterStart.disabled = running;
    this.btnPlotterStop.disabled = !running;
    if (this.plotStartBtn) this.plotStartBtn.disabled = running;
    if (this.plotStopBtn) this.plotStopBtn.disabled = !running;
  }

  async onPlotterStart(): Promise<void> {
    const encOpt = this.plotterEncoderFpsStartOption();
    if (encOpt !== undefined && (encOpt < 1 || encOpt > 50)) {
      this.setPlaybackStatus(
        "Encoder FPS must be between 1 and 50, or leave the field empty for the server default.",
      );
      return;
    }
    this.btnPlotterStart.disabled = true;
    if (this.plotStartBtn) this.plotStartBtn.disabled = true;
    try {
      const startBody: { encoder_fps?: number } = {};
      if (encOpt !== undefined) startBody.encoder_fps = encOpt;
      let j: { status: string; encoder_fps?: number | null };
      const r = await this.api.postPlotterStart(
        Object.keys(startBody).length ? startBody : undefined,
        logPlayback,
      );
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
      }
      j = (await r.json()) as { status: string; encoder_fps?: number | null };
      if (j.status === "started" || j.status === "already_running") {
        this.syncPlotterEncoderFpsInputFromResponse(j);
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

  async onGeneratorStart(): Promise<void> {
    this.btnGeneratorStart.disabled = true;
    const uiMode = normalizeUiMode(this.selectedMode());
    try {
      const r = await this.api.postGeneratorStart(this.buildGeneratorStartBody(uiMode), logPlayback);
      if (!r.ok) {
        const t = await r.text();
        throw new Error(this.parseApiErrorDetail(t) || r.statusText);
      }
      const j = (await r.json()) as { status: string };
      if (j.status === "started") {
        this.lastGenerating = false;
      }
      if (uiMode === SYNTHETIC_IMAGES) {
        this.setPlaybackStatus(
          "Video generator started — connecting… If you use LL-HLS, the first manifest requests may 404 until RTSP publish is live.",
        );
      } else if (uiMode === SYNTHETIC_RESOURCE_GRID) {
        this.setPlaybackStatus("SYNTHETIC_RESOURCE_GRID generator started.");
      } else {
        this.setPlaybackStatus("SYNTHETIC_FAPI generator started.");
      }
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
    const uiMode = normalizeUiMode(this.selectedMode());
    try {
      const r = await this.api.postGeneratorFeedStop(logPlayback);
      if (!r.ok) {
        const t = await r.text();
        throw new Error(this.parseApiErrorDetail(t) || r.statusText);
      }
      const j = (await r.json()) as { status: string };
      if (j.status === "not_running") {
        this.setPlaybackStatus("Generator was not running.");
      } else if (uiMode === SYNTHETIC_IMAGES) {
        this.setPlaybackStatus("Video generator stopped.");
      } else {
        this.setPlaybackStatus("SYNTHETIC_RESOURCE_GRID / SYNTHETIC_FAPI generator stopped.");
      }
      await this.refreshPublisherStatus();
    } catch (e) {
      this.setPlaybackStatus(
        `Generator stop failed: ${e instanceof Error ? e.message : String(e)}`,
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
        const st = await this.fetchPublisherStatus();
        if (!st) return;
        this.applyPublisherStatus(st);
        if (!st.capturer_active) {
          if (this.capturerPoll) clearInterval(this.capturerPoll);
          this.capturerPoll = null;
        }
      })();
    }, 400);
  }

  async onPlotterStop(): Promise<void> {
    this.btnPlotterStop.disabled = true;
    if (this.plotStopBtn) this.plotStopBtn.disabled = true;
    try {
      const r = await this.api.postPlotterStop(logPlayback);
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || r.statusText);
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

  private parseApiErrorDetail(body: string): string {
    const raw = body.trim();
    if (!raw) return "Request failed";
    try {
      const j = JSON.parse(raw) as { detail?: unknown };
      if (typeof j.detail === "string") return j.detail;
      if (Array.isArray(j.detail)) {
        return j.detail
          .map((item: unknown) =>
            item && typeof item === "object" && item !== null && "msg" in (item as object)
              ? String((item as { msg: string }).msg)
              : String(item),
          )
          .join("; ");
      }
    } catch {
      /* not JSON */
    }
    return raw;
  }

  private setSwaggerLink(): void {
    const swagger = queryRole<HTMLAnchorElement>(this.controlsRoot, "swagger-link");
    if (swagger) swagger.href = this.api.swaggerDocsHref();
  }

  start(): void {
    this.initModeSelect();
    queryRole<HTMLSelectElement>(this.controlsRoot, "mode-select")?.addEventListener("change", () => {
      this.updatePanelVisibilityForMode();
      if (this.lastPublisherStatus) this.applyPublisherStatus(this.lastPublisherStatus);
    });
    this.setSwaggerLink();
    this.btnPlotterStart.addEventListener("click", () => void this.onPlotterStart());
    this.btnPlotterStop.addEventListener("click", () => void this.onPlotterStop());
    this.plotStartBtn?.addEventListener("click", () => void this.onPlotterStart());
    this.plotStopBtn?.addEventListener("click", () => void this.onPlotterStop());
    this.btnGeneratorStart.addEventListener("click", () => void this.onGeneratorStart());
    this.btnGeneratorStop.addEventListener("click", () => void this.onGeneratorStop());
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-generator-resource-grid-fps-apply")?.addEventListener(
      "click",
      () => void this.putIqGeneratorFps(),
    );
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-start")?.addEventListener("click", () =>
      void this.onCapturerStart(),
    );
    queryRole<HTMLButtonElement>(this.controlsRoot, "btn-capturer-stop")?.addEventListener("click", () =>
      void this.onCapturerStop(),
    );

    this.playback.attachVideoListeners();

    this.statusPoll = setInterval(() => void this.refreshPublisherStatus(), PUBLISHER_STATUS_POLL_MS);

    logPlayback("boot", {
      href: typeof window !== "undefined" ? window.location.href : "",
      apiBase: this.api.baseUrl,
      playback: playbackMode(),
      debug: debugEnabled,
    });
    void (async () => {
      await this.loadIqChannelLegend();
      this.populateLavfiSourceSelect();
      this.updatePanelVisibilityForMode();
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
