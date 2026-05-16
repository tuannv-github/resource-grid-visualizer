import { axisTickValues } from "../lib/format";
import { queryRole } from "./dom";

/**
 * Symbol along horizontal axis (equal-width columns); PRB along vertical (PRB 0 at bottom; ticks use column-reverse CSS).
 *
 * When `symbolStart` / `symbolStop` are supplied (server **iq_symbol_start** / **iq_symbol_stop**, both inclusive),
 * the axis only renders ticks for symbols in that window so it stays aligned with the cropped raster. Defaults
 * (`undefined` or full slot) render every symbol from `0` through `symCount - 1`.
 */
export function renderIqAxes(
  root: Element,
  prbCount: number,
  symCount: number,
  iqMode: boolean,
  symbolStart?: number,
  symbolStop?: number,
): void {
  const videoAxes = queryRole<HTMLElement>(root, "video-axes");
  if (!videoAxes) return;

  videoAxes.dataset.iqVisible = iqMode ? "true" : "false";
  if (!iqMode) return;

  const prbCol = queryRole<HTMLElement>(root, "va-symbol-ticks");
  const symFoot = queryRole<HTMLElement>(root, "va-axis-bottom");
  if (!prbCol || !symFoot) return;

  const prbMax = Math.max(0, prbCount - 1);
  const totalSymbols = Math.max(1, symCount);
  const fullLastIdx = totalSymbols - 1;
  const rawStart = Number.isFinite(symbolStart) ? Math.trunc(symbolStart as number) : 0;
  const rawStop = Number.isFinite(symbolStop) ? Math.trunc(symbolStop as number) : fullLastIdx;
  const clampedStart = Math.min(Math.max(0, rawStart), fullLastIdx);
  const clampedStop = Math.min(Math.max(clampedStart, rawStop), fullLastIdx);
  const visibleCount = Math.max(1, clampedStop - clampedStart + 1);
  /* Cropped means the plotter rendered a sub-window of the slot (e.g. START_SYMBOL=2 / STOP_SYMBOL=11).
   * In that case the simple "0..N-1" symbol ticks no longer align with the raster columns, so hide the
   * symbol axis chrome (top label, bottom ticks, top-left corner) — PRB ticks on the left stay because
   * they still map 1:1 to the cropped image rows. */
  const cropped = clampedStart !== 0 || clampedStop !== fullLastIdx;
  videoAxes.dataset.iqSymbolAxisCropped = cropped ? "true" : "false";

  prbCol.replaceChildren();
  for (const v of axisTickValues(prbMax, 7)) {
    const span = document.createElement("span");
    span.textContent = String(v);
    prbCol.appendChild(span);
  }
  symFoot.replaceChildren();
  if (cropped) {
    /* Still set aria-label so screen readers describe the cropped window even though the ticks are
     * visually hidden — keeps the axis discoverable for assistive tech / debugging. */
    symFoot.setAttribute(
      "aria-label",
      `Symbol axis hidden; plotter is rendering symbols ${clampedStart}–${clampedStop} of ${fullLastIdx + 1} per slot`,
    );
    return;
  }
  symFoot.style.gridTemplateColumns = `repeat(${visibleCount}, minmax(0, 1fr))`;
  symFoot.setAttribute("aria-label", `Symbol index 0 through ${fullLastIdx}`);
  for (let i = 0; i < visibleCount; i++) {
    const span = document.createElement("span");
    span.textContent = String(clampedStart + i);
    symFoot.appendChild(span);
  }
}
