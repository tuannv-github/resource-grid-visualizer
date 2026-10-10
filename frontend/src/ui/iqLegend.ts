import type { IqChannelLegendPayload, MagScaleLegend } from "../domain/publisherStatus";
import { plainLegendNote } from "../lib/format";
import { queryRole } from "./dom";

function formatMag(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(1);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/** Vertical Viridis bar for the fixed linear-magnitude scale (`0` … `mag_scale`). */
export function renderMagnitudeScale(root: Element, magnitude: MagScaleLegend | undefined): void {
  const host = queryRole<HTMLElement>(root, "va-mag-scale");
  const bar = queryRole<HTMLElement>(root, "va-mag-bar");
  const ticks = queryRole<HTMLElement>(root, "va-mag-ticks");
  const caption = queryRole<HTMLElement>(root, "va-mag-caption");
  if (!host || !bar || !ticks) return;

  const stops = magnitude?.stops?.filter((s) => Number.isFinite(s.t) && typeof s.hex === "string") ?? [];
  const vmin = magnitude?.vmin;
  const vmax = magnitude?.vmax;
  if (
    !magnitude ||
    stops.length < 2 ||
    vmin == null ||
    vmax == null ||
    !Number.isFinite(vmin) ||
    !Number.isFinite(vmax) ||
    !(vmax > vmin)
  ) {
    host.hidden = true;
    if (caption) caption.hidden = true;
    return;
  }

  const ordered = [...stops].sort((a, b) => a.t - b.t);
  bar.style.backgroundImage = `linear-gradient(to top, ${ordered
    .map((s) => `${s.hex} ${(s.t * 100).toFixed(2)}%`)
    .join(", ")})`;
  const label = `Linear magnitude |I+jQ|, ${formatMag(vmin)} to ${formatMag(vmax)}. Values above the top stay at the top color.`;
  host.setAttribute("aria-label", label);
  bar.title = label;

  ticks.replaceChildren();
  for (const frac of [0, 0.5, 1]) {
    const span = document.createElement("span");
    span.textContent = formatMag(vmin + (vmax - vmin) * frac);
    span.style.bottom = `${frac * 100}%`;
    ticks.appendChild(span);
  }

  host.hidden = false;
  if (caption) caption.hidden = false;
}

/** Logical ids from OAI / plotter pairing: DL 1,3,5,7 · UL 2,4,6,8 */
const LEGEND_DL_IDS = new Set([1, 3, 5, 7]);
const LEGEND_UL_IDS = new Set([2, 4, 6, 8]);

function appendLegendRows(list: HTMLElement, channels: IqChannelLegendPayload["channels"]): void {
  list.replaceChildren();
  for (const ch of channels) {
    const li = document.createElement("li");
    li.className = "iq-legend-item";

    const sw = document.createElement("span");
    sw.className = "iq-legend-swatch";
    sw.style.backgroundColor = ch.hex;
    sw.title = `RGB ${ch.rgb.r}, ${ch.rgb.g}, ${ch.rgb.b}`;
    sw.setAttribute("role", "img");
    sw.setAttribute("aria-label", `${ch.label} color`);

    const lab = document.createElement("span");
    lab.className = "iq-legend-label";
    lab.textContent = ch.label;

    li.append(sw, lab);
    list.appendChild(li);
  }
}

export function renderIqChannelLegend(root: Element, payload: IqChannelLegendPayload): void {
  const dlList = queryRole<HTMLElement>(root, "iq-legend-list-dl");
  const ulList = queryRole<HTMLElement>(root, "iq-legend-list-ul");
  const notes = queryRole<HTMLElement>(root, "iq-legend-notes");
  if (!dlList || !ulList || !notes) return;

  const dlChannels = payload.channels.filter((c) => LEGEND_DL_IDS.has(c.id)).sort((a, b) => a.id - b.id);
  const ulChannels = payload.channels.filter((c) => LEGEND_UL_IDS.has(c.id)).sort((a, b) => a.id - b.id);

  appendLegendRows(dlList, dlChannels);
  appendLegendRows(ulList, ulChannels);

  notes.replaceChildren();
  const n0 = plainLegendNote(payload.channel_zero_note);
  const n1 = plainLegendNote(payload.render_note);
  if (n0) {
    const p0 = document.createElement("p");
    p0.textContent = n0;
    notes.appendChild(p0);
  }
  if (n1) {
    const p1 = document.createElement("p");
    p1.textContent = n1;
    notes.appendChild(p1);
  }
}

export function renderIqLegendError(root: Element, message: string): void {
  const dlList = queryRole<HTMLElement>(root, "iq-legend-list-dl");
  const ulList = queryRole<HTMLElement>(root, "iq-legend-list-ul");
  const notes = queryRole<HTMLElement>(root, "iq-legend-notes");
  if (!dlList || !ulList || !notes) return;
  dlList.replaceChildren();
  ulList.replaceChildren();
  notes.replaceChildren();
  const p = document.createElement("p");
  p.className = "iq-legend-error-msg";
  p.textContent = message;
  notes.appendChild(p);
}
