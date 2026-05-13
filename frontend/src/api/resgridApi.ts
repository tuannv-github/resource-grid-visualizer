import type {
  IqChannelLegendPayload,
  PublisherStatus,
} from "../domain/publisherStatus";
import type { GeneratorFeedStartMode, GeneratorStartBody } from "../domain/generatorModes";

export class ResgridApi {
  readonly baseUrl: string;

  constructor(apiBase: string) {
    this.baseUrl = apiBase.replace(/\/$/, "");
  }

  async fetchPublisherStatus(log?: (...a: unknown[]) => void): Promise<PublisherStatus | null> {
    const plotterUrl = `${this.baseUrl}/api/plotter/status`;
    const generatorUrl = `${this.baseUrl}/api/generator/status`;
    try {
      const [rp, rg] = await Promise.all([fetch(plotterUrl), fetch(generatorUrl)]);
      log?.("GET", plotterUrl, "→", rp.status, rp.ok ? "ok" : rp.statusText);
      log?.("GET", generatorUrl, "→", rg.status, rg.ok ? "ok" : rg.statusText);
      if (!rp.ok) return null;
      const p = (await rp.json()) as Record<string, unknown>;
      let g: Record<string, unknown> | null = null;
      if (rg.ok) {
        try {
          g = (await rg.json()) as Record<string, unknown>;
        } catch {
          g = null;
        }
      }
      const syn = (g?.synthetic_video as Record<string, unknown> | undefined) ?? {};
      const srg = (g?.synthetic_resource_grid as Record<string, unknown> | undefined) ?? {};
      const sfa = (g?.synthetic_fapi as Record<string, unknown> | undefined) ?? {};
      const mode = typeof g?.generation_mode === "string" ? (g.generation_mode as string) : "";
      const lavfi =
        syn.lavfi_video === null || syn.lavfi_video === undefined
          ? null
          : String(syn.lavfi_video);
      const iqRun = srg.running === true || sfa.running === true;
      const iqCap = srg.capable === true || sfa.capable === true;
      const fifoPath = (v: unknown): string | null =>
        v == null ? null : typeof v === "string" ? v : String(v);
      const fifoWriterPid = (v: unknown): number | null => {
        if (v == null) return null;
        const n = typeof v === "number" ? v : Number(v);
        return Number.isFinite(n) ? Math.trunc(n) : null;
      };
      const j = {
        ...p,
        mode,
        lavfi_video: lavfi,
        video_generator_source: syn.source ?? null,
        video_generator_running: syn.running === true,
        video_generator_fifo_path: fifoPath(syn.fifo_path),
        iq_generator_running: iqRun,
        iq_generator_capable: iqCap,
        synthetic_resource_grid: g
          ? {
              running: srg.running === true,
              capable: srg.capable === true,
              pid: fifoWriterPid(srg.pid),
              fifo_path: fifoPath(srg.fifo_path),
            }
          : undefined,
        synthetic_fapi: g
          ? {
              running: sfa.running === true,
              capable: sfa.capable === true,
              pid: fifoWriterPid(sfa.pid),
              fifo_path: fifoPath(sfa.fifo_path),
            }
          : undefined,
      } as PublisherStatus;
      log?.("status payload", {
        generating: j.generating,
        mtx_path: j.mtx_path,
        mode: j.mode,
        encoder: j.video_encoder,
      });
      return j;
    } catch (e) {
      log?.("GET", plotterUrl, "/", generatorUrl, "failed (network / CORS?)", e);
      return null;
    }
  }

  async getIqChannelLegend(
    log?: (...a: unknown[]) => void,
  ): Promise<
    | { ok: true; payload: IqChannelLegendPayload }
    | { ok: false; message: string }
  > {
    const url = `${this.baseUrl}/api/generator/channel-legend`;
    try {
      const r = await fetch(url);
      log?.("GET", url, "→", r.status, r.ok ? "ok" : r.statusText);
      if (!r.ok) {
        return { ok: false, message: `Legend unavailable (${r.status}).` };
      }
      return { ok: true, payload: (await r.json()) as IqChannelLegendPayload };
    } catch (e) {
      log?.("GET", url, "failed", e);
      return { ok: false, message: "Legend unavailable (network error)." };
    }
  }

  async putIqGeneratorFps(fps: number, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/generator/fps`;
    const r = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fps }),
    });
    log?.("PUT", url, "→", r.status, r.ok ? "ok" : r.statusText);
    return r;
  }

  /** **POST /api/generator/start** with **`feed`**: **images** | **resource_grid** | **fapi**. */
  async postGeneratorStart(
    body: GeneratorStartBody,
    log?: (...a: unknown[]) => void,
  ): Promise<Response> {
    const url = `${this.baseUrl}/api/generator/start`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    log?.("POST", url, body, "→", r.status);
    return r;
  }

  async postGeneratorFeedStop(log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/generator/stop`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    log?.("POST", url, "→", r.status);
    return r;
  }

  /** Stop **FFmpeg** and/or **CUDA plotter** (**POST /api/stop**). Optional body symmetry for clients. */
  async postStopAll(mode: string, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stop`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    log?.("POST", url, mode, "→", r.status);
    return r;
  }

  /** Plotter spawn: optional **encoder_fps** only (**POST /api/plotter/start** does not take **`mode`**). */
  async postPlotterStart(
    opts?: { encoder_fps?: number },
    log?: (...a: unknown[]) => void,
  ): Promise<Response> {
    const url = `${this.baseUrl}/api/plotter/start`;
    const body: { encoder_fps?: number } = {};
    if (opts?.encoder_fps !== undefined) body.encoder_fps = opts.encoder_fps;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    log?.("POST", url, body, "→", r.status);
    return r;
  }

  async postPlotterStop(log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/plotter/stop`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    log?.("POST", url, "→", r.status);
    return r;
  }

  async postPlotterCapturer(
    frames: number,
    log?: (...a: unknown[]) => void,
  ): Promise<Response> {
    const url = `${this.baseUrl}/api/plotter/capturer`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    log?.("POST", url, frames, "→", r.status);
    return r;
  }

  async postPlotterCapturerStop(log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/plotter/capturer/stop`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    log?.("POST", url, "→", r.status);
    return r;
  }

  /**
   * **POST /api/generator/start** with **`{ "feed": "resource_grid" }`** or **stop**.
   * Prefer **`postGeneratorStart`** / **`postGeneratorFeedStop`** for feed control.
   */
  async postGenerator(path: "start" | "stop", log?: (...a: unknown[]) => void): Promise<Response> {
    const url =
      path === "start" ? `${this.baseUrl}/api/generator/start` : `${this.baseUrl}/api/generator/stop`;
    const init: RequestInit =
      path === "start"
        ? {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feed: "resource_grid" }),
          }
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          };
    const r = await fetch(url, init);
    log?.("POST", url, "→", r.status);
    return r;
  }

  swaggerDocsHref(): string {
    return `${this.baseUrl}/docs`;
  }
}
