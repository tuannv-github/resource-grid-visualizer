import type { IqChannelLegendPayload } from "../domain/publisherStatus";
import { plainLegendNote } from "../lib/format";
import { queryRole } from "./dom";

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
