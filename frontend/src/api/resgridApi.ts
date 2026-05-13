import type {
  IqChannelLegendPayload,
  IqPatternCatalog,
  StreamStatus,
} from "../domain/streamTypes";

export class ResgridApi {
  readonly baseUrl: string;

  constructor(apiBase: string) {
    this.baseUrl = apiBase.replace(/\/$/, "");
  }

  async fetchStreamStatus(log?: (...a: unknown[]) => void): Promise<StreamStatus | null> {
    const url = `${this.baseUrl}/api/stream/status`;
    try {
      const r = await fetch(url);
      log?.("GET", url, "→", r.status, r.ok ? "ok" : r.statusText);
      if (!r.ok) return null;
      const j = (await r.json()) as StreamStatus;
      log?.("status payload", {
        generating: j.generating,
        mtx_path: j.mtx_path,
        stream_mode: j.stream_mode,
        encoder: j.video_encoder,
      });
      return j;
    } catch (e) {
      log?.("GET", url, "failed (network / CORS?)", e);
      return null;
    }
  }

  async fetchIqPatternCatalog(): Promise<IqPatternCatalog | null> {
    const url = `${this.baseUrl}/api/stream/iq/patterns`;
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      return (await r.json()) as IqPatternCatalog;
    } catch {
      return null;
    }
  }

  async getIqChannelLegend(
    log?: (...a: unknown[]) => void,
  ): Promise<
    | { ok: true; payload: IqChannelLegendPayload }
    | { ok: false; message: string }
  > {
    const url = `${this.baseUrl}/api/stream/iq/channel-legend`;
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

  async putIqPattern(pattern: number, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/iq/pattern`;
    const r = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pattern }),
    });
    log?.("PUT", url, "→", r.status, r.ok ? "ok" : r.statusText);
    return r;
  }

  async putIqGeneratorFps(fps: number, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/iq/generator-fps`;
    const r = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fps }),
    });
    log?.("PUT", url, "→", r.status, r.ok ? "ok" : r.statusText);
    return r;
  }

  async postStream(path: "start" | "stop", mode: string, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/${path}`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    log?.("POST", url, mode, "→", r.status);
    return r;
  }

  async postPlotterStart(mode: string, log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/plotter/start`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    log?.("POST", url, mode, "→", r.status);
    return r;
  }

  async postPlotterStop(log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/plotter/stop`;
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
    const url = `${this.baseUrl}/api/stream/plotter/capturer`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    log?.("POST", url, frames, "→", r.status);
    return r;
  }

  async postPlotterCapturerStop(log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/plotter/capturer/stop`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    log?.("POST", url, "→", r.status);
    return r;
  }

  async postGenerator(path: "start" | "stop", log?: (...a: unknown[]) => void): Promise<Response> {
    const url = `${this.baseUrl}/api/stream/generator/${path}`;
    const r = await fetch(url, { method: "POST" });
    log?.("POST", url, "→", r.status);
    return r;
  }

  swaggerDocsHref(): string {
    return `${this.baseUrl}/docs`;
  }
}
