import { isPlaybackDebugEnabled } from "../config/debug";
import { resolvePanels, type ResolvedGuiLayout } from "../config/panelLayout";
import { ResgridPanel } from "../panel/ResgridPanel";
import { mountSidebarToggle } from "../ui/sidebarToggle";

const livePanels: ResgridPanel[] = [];

const ALL_MEDIA_IDS = ["panel-dl-media", "panel-ul-media", "panel-nai-media"] as const;
const ALL_CONTROLS_IDS = ["panel-dl-controls", "panel-ul-controls", "panel-nai-controls"] as const;

function ensureNaiPanelsExist(): void {
  const naiMediaExists = Boolean(document.getElementById("panel-nai-media"));
  const naiControlsExists = Boolean(document.getElementById("panel-nai-controls"));
  if (naiMediaExists && naiControlsExists) return;

  const ulMedia = document.getElementById("panel-ul-media");
  const ulControls = document.getElementById("panel-ul-controls");
  const mediaStack = document.getElementById("rg-media-stack");
  if (!ulMedia || !ulControls || !mediaStack) return;

  if (!naiMediaExists) {
    const naiMedia = ulMedia.cloneNode(true) as HTMLElement;
    naiMedia.id = "panel-nai-media";
    naiMedia.hidden = false;
    naiMedia.removeAttribute("hidden");
    naiMedia.setAttribute("data-direction", "nai");
    naiMedia.setAttribute("aria-label", "Noise and Interference Isolation video");
    mediaStack.appendChild(naiMedia);
  }

  if (!naiControlsExists) {
    const naiControls = ulControls.cloneNode(true) as HTMLElement;
    naiControls.id = "panel-nai-controls";
    naiControls.hidden = false;
    naiControls.removeAttribute("hidden");
    const side = document.querySelector(".rg-dashboard__side");
    if (side) side.appendChild(naiControls);
    else ulControls.parentElement?.appendChild(naiControls);
  }
}

/** Drop panels that are not part of this layout (rxnai: no DL; dlul: no NAI). */
function prunePanelsNotInLayout(desiredMedia: Set<string>, desiredControls: Set<string>): void {
  for (const id of ALL_MEDIA_IDS) {
    if (!desiredMedia.has(id)) document.getElementById(id)?.remove();
  }
  for (const id of ALL_CONTROLS_IDS) {
    if (!desiredControls.has(id)) document.getElementById(id)?.remove();
  }
}

function applyLayoutRootClasses(app: HTMLElement, cfg: ResolvedGuiLayout): void {
  app.classList.toggle("rg-app--dual", cfg.dual);
  app.classList.remove("rg-app--dlul", "rg-app--rxnai", "rg-app--ulnai", "rg-app--combine");
  if (cfg.layoutMode === "dlul") app.classList.add("rg-app--dlul");
  else if (cfg.layoutMode === "rxnai" || cfg.layoutMode === "ulnai") {
    app.classList.add("rg-app--rxnai", "rg-app--ulnai");
  }
  else if (cfg.layoutMode === "combine") app.classList.add("rg-app--combine");
}

function syncDomToLayout(cfg: ResolvedGuiLayout): void {
  const desiredMedia = new Set(cfg.panels.map((p) => p.id));
  const desiredControls = new Set(cfg.panels.map((p) => p.controlsRootId));

  if (desiredMedia.has("panel-nai-media")) {
    ensureNaiPanelsExist();
  }

  prunePanelsNotInLayout(desiredMedia, desiredControls);

  for (const id of desiredMedia) {
    const el = document.getElementById(id);
    if (el) {
      el.hidden = false;
      el.removeAttribute("hidden");
    }
  }
  for (const id of desiredControls) {
    const el = document.getElementById(id);
    if (el) {
      el.hidden = false;
      el.removeAttribute("hidden");
    }
  }
}

export function bootApp(): void {
  const cfg = resolvePanels();
  const app = document.getElementById("app");
  if (!app) return;

  applyLayoutRootClasses(app, cfg);
  syncDomToLayout(cfg);
  /* Mount before panels: applies persisted `rg-app--sidebar-hidden` synchronously so the dashboard
   * grid mounts in its final shape (no flash of full sidebar on reload). */
  mountSidebarToggle(app);

  console.info(
    "[resgrid] layout=%s panels=%s",
    cfg.layoutMode,
    cfg.panels.map((p) => `${p.id}→${p.playbackPath}`).join(", "),
  );

  for (const spec of cfg.panels) {
    const mediaRoot = document.getElementById(spec.id);
    const controlsRoot = document.getElementById(spec.controlsRootId);
    if (!mediaRoot || !controlsRoot) {
      console.warn("[resgrid] missing DOM for panel", spec.id, spec.controlsRootId);
      continue;
    }
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
