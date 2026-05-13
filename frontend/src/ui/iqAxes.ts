import { axisTickValues } from "../lib/format";
import { queryRole } from "./dom";

/** Symbol along horizontal axis (equal-width columns); PRB along vertical (PRB 0 at bottom; ticks use column-reverse CSS). */
export function renderIqAxes(
  root: Element,
  prbCount: number,
  symCount: number,
  iqMode: boolean,
): void {
  const videoAxes = queryRole<HTMLElement>(root, "video-axes");
  if (!videoAxes) return;

  videoAxes.dataset.iqVisible = iqMode ? "true" : "false";
  if (!iqMode) return;

  const prbCol = queryRole<HTMLElement>(root, "va-symbol-ticks");
  const symFoot = queryRole<HTMLElement>(root, "va-axis-bottom");
  if (!prbCol || !symFoot) return;

  const prbMax = Math.max(0, prbCount - 1);
  const symbolColumnCount = Math.max(1, symCount);
  const lastSymbolIdx = symbolColumnCount - 1;

  prbCol.replaceChildren();
  for (const v of axisTickValues(prbMax, 7)) {
    const span = document.createElement("span");
    span.textContent = String(v);
    prbCol.appendChild(span);
  }
  symFoot.replaceChildren();
  symFoot.style.gridTemplateColumns = `repeat(${symbolColumnCount}, minmax(0, 1fr))`;
  symFoot.setAttribute("aria-label", `Symbol index 0 through ${lastSymbolIdx}`);
  for (let symIdx = 0; symIdx < symbolColumnCount; symIdx++) {
    const span = document.createElement("span");
    span.textContent = String(symIdx);
    symFoot.appendChild(span);
  }
}
