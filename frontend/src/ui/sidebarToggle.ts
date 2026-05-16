/**
 * Toolbar control that hides / shows the dashboard sidebar (`.rg-dashboard__side`).
 *
 * Adds the `rg-app--sidebar-hidden` modifier on `#app` to collapse the dashboard grid to a single
 * media column. The user choice is persisted to `localStorage` under `SIDEBAR_HIDDEN_STORAGE_KEY`
 * so it survives reloads and is applied **before** panels boot (no layout flash).
 *
 * DOM contract (see `index.html`):
 *   - `<button data-role="btn-sidebar-toggle" aria-controls="rg-sidebar" aria-pressed="…">`
 *     with a child `[data-role="btn-sidebar-toggle-label"]` (text changes between "Hide sidebar"
 *     and "Show sidebar"). `aria-pressed="true"` ↔ sidebar is hidden.
 */

const SIDEBAR_HIDDEN_STORAGE_KEY = "resgrid:sidebarHidden";
const SIDEBAR_HIDDEN_CLASS = "rg-app--sidebar-hidden";
const LABEL_HIDDEN = "Show sidebar";
const LABEL_VISIBLE = "Hide sidebar";

function safeReadHidden(): boolean {
  try {
    return globalThis.localStorage?.getItem(SIDEBAR_HIDDEN_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function safeWriteHidden(hidden: boolean): void {
  try {
    globalThis.localStorage?.setItem(SIDEBAR_HIDDEN_STORAGE_KEY, hidden ? "1" : "0");
  } catch {
    /* private mode / disabled storage — toggle still works in-memory */
  }
}

/** Apply the persisted preference to `#app` synchronously so the dashboard mounts in the right shape. */
export function applyPersistedSidebarState(app: HTMLElement): boolean {
  const hidden = safeReadHidden();
  app.classList.toggle(SIDEBAR_HIDDEN_CLASS, hidden);
  return hidden;
}

function syncButtonState(button: HTMLButtonElement, hidden: boolean): void {
  button.setAttribute("aria-pressed", hidden ? "true" : "false");
  const labelEl = button.querySelector<HTMLElement>('[data-role="btn-sidebar-toggle-label"]');
  if (labelEl) labelEl.textContent = hidden ? LABEL_HIDDEN : LABEL_VISIBLE;
  button.title = hidden
    ? "Show the Plotter & Generator side panel"
    : "Hide the Plotter & Generator side panel";
}

/** Bind the toolbar button so clicks toggle the sidebar; returns the initial hidden state. */
export function mountSidebarToggle(app: HTMLElement): boolean {
  const initiallyHidden = applyPersistedSidebarState(app);
  const button = document.querySelector<HTMLButtonElement>('[data-role="btn-sidebar-toggle"]');
  if (!button) return initiallyHidden;
  syncButtonState(button, initiallyHidden);
  button.addEventListener("click", () => {
    const nowHidden = !app.classList.contains(SIDEBAR_HIDDEN_CLASS);
    app.classList.toggle(SIDEBAR_HIDDEN_CLASS, nowHidden);
    safeWriteHidden(nowHidden);
    syncButtonState(button, nowHidden);
  });
  return initiallyHidden;
}
