import { isPlaybackDebugEnabled } from "../config/debug";
import { resolvePanels } from "../config/panelLayout";
import { ResgridPanel } from "../panel/ResgridPanel";

const livePanels: ResgridPanel[] = [];

export function bootApp(): void {
  const cfg = resolvePanels();
  const app = document.getElementById("app");
  if (!app) return;
  app.classList.toggle("rg-app--dual", cfg.dual);

  const ulMedia = document.getElementById("panel-ul-media");
  const ulControls = document.getElementById("panel-ul-controls");
  if (!cfg.dual) {
    /* Combine / single-API: one stream — drop hidden UL markup (no second va-video-wrap / player). */
    ulMedia?.remove();
    ulControls?.remove();
  } else {
    if (ulMedia) ulMedia.hidden = false;
    if (ulControls) ulControls.hidden = false;
  }

  for (const spec of cfg.panels) {
    const mediaRoot = document.getElementById(spec.id);
    const controlsRoot = document.getElementById(spec.controlsRootId);
    if (!mediaRoot || !controlsRoot) continue;
    const heading = controlsRoot.querySelector<HTMLElement>(".rg-panel-heading");
    if (heading) heading.textContent = spec.heading;
    const panel = new ResgridPanel(mediaRoot as HTMLElement, {
      apiBase: spec.apiBase,
      playbackPath: spec.playbackPath,
      modeSelectId: spec.modeSelectId,
      controlsRoot: controlsRoot as HTMLElement,
    });
    panel.start();
    livePanels.push(panel);
  }

  console.info(
    "[resgrid] App loaded. “Unchecked runtime.lastError…” is from a browser extension, not MediaMTX.",
  );
  if (isPlaybackDebugEnabled()) {
    console.info(
      "[resgrid] Verbose traces: add ?debug=1 to the URL, filter console by “resgrid”. layout =",
      cfg.layoutMode,
      "| dual =",
      cfg.dual,
      "| panels =",
      cfg.panels.map((p) => `${p.id}→${p.apiBase}`).join(", "),
    );
  }

  window.addEventListener("beforeunload", () => {
    for (const p of livePanels) p.destroy();
  });
}
