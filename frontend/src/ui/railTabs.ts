/**
 * Two rail screens share the same plots:
 *   grids    — video only (sidebar stays off)
 *   generate — the plotter and generator controls, for generation tests
 */

const SIDEBAR_HIDDEN_CLASS = "rg-app--sidebar-hidden";

type RailTab = "grids" | "generate";

function syncToggle(visibleSidebar: boolean): void {
  const button = document.querySelector<HTMLButtonElement>('[data-role="btn-sidebar-toggle"]');
  if (!button) return;
  button.hidden = !visibleSidebar;
  if (!visibleSidebar) return;
  button.setAttribute("aria-pressed", "false");
  const labelEl = button.querySelector<HTMLElement>('[data-role="btn-sidebar-toggle-label"]');
  if (labelEl) labelEl.textContent = "Hide sidebar";
  button.title = "Hide the Plotter and Generator side panel";
}

function showTab(app: HTMLElement, tab: RailTab): void {
  const generate = tab === "generate";
  app.classList.toggle("rg-app--tab-generate", generate);
  app.classList.toggle(SIDEBAR_HIDDEN_CLASS, !generate);
  document.querySelectorAll<HTMLButtonElement>("[data-rail-tab]").forEach((button) => {
    const on = button.dataset.railTab === tab;
    button.classList.toggle("rail-tab-active", on);
    if (on) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  const crumb = document.querySelector<HTMLElement>(".crumb");
  if (crumb) crumb.textContent = generate ? "Generation test" : "Resource grid";
  syncToggle(generate);
}

export function mountRailTabs(app: HTMLElement): void {
  document.querySelectorAll<HTMLButtonElement>("[data-rail-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.railTab === "generate" ? "generate" : "grids";
      showTab(app, tab);
    });
  });
  showTab(app, "grids");
}
