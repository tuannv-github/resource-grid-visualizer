/**
 * Toolbar control that hides / shows the dashboard sidebar on the Generate tab.
 *
 * DOM contract (see `index.html`):
 *   - `<button data-role="btn-sidebar-toggle" aria-controls="rg-sidebar" aria-pressed="…">`
 *     with a child `[data-role="btn-sidebar-toggle-label"]`.
 *     `aria-pressed="true"` means the sidebar is hidden.
 */

const SIDEBAR_HIDDEN_CLASS = "rg-app--sidebar-hidden";
const LABEL_HIDDEN = "Show sidebar";
const LABEL_VISIBLE = "Hide sidebar";

function syncButtonState(button: HTMLButtonElement, hidden: boolean): void {
  button.setAttribute("aria-pressed", hidden ? "true" : "false");
  const labelEl = button.querySelector<HTMLElement>('[data-role="btn-sidebar-toggle-label"]');
  if (labelEl) labelEl.textContent = hidden ? LABEL_HIDDEN : LABEL_VISIBLE;
  button.title = hidden
    ? "Show the Plotter and Generator side panel"
    : "Hide the Plotter and Generator side panel";
}

/** Bind the toolbar button so clicks toggle the sidebar on the Generate tab. */
export function mountSidebarToggle(app: HTMLElement): boolean {
  const button = document.querySelector<HTMLButtonElement>('[data-role="btn-sidebar-toggle"]');
  if (!button) return app.classList.contains(SIDEBAR_HIDDEN_CLASS);
  syncButtonState(button, app.classList.contains(SIDEBAR_HIDDEN_CLASS));
  button.addEventListener("click", () => {
    if (!app.classList.contains("rg-app--tab-generate")) return;
    const nowHidden = !app.classList.contains(SIDEBAR_HIDDEN_CLASS);
    app.classList.toggle(SIDEBAR_HIDDEN_CLASS, nowHidden);
    syncButtonState(button, nowHidden);
  });
  return app.classList.contains(SIDEBAR_HIDDEN_CLASS);
}
