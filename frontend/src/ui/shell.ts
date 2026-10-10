/**
 * NeuroRAN shell chrome: site badge, local clock, and dark/light theme.
 * Accent values match the Energy green palette (rApp default).
 */

const THEME_KEY = "resgrid:theme";

const ICON_SUN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path></svg>`;
const ICON_MOON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;

function readDark(): boolean {
  try {
    return localStorage.getItem(THEME_KEY) !== "light";
  } catch {
    return true;
  }
}

function writeDark(dark: boolean): void {
  try {
    localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
  } catch {
    /* private mode */
  }
}

function applyTheme(app: HTMLElement, dark: boolean): void {
  app.classList.toggle("theme-dark", dark);
  app.classList.toggle("theme-light", !dark);
  if (dark) {
    app.style.setProperty("--accent", "#16E6A0");
    app.style.setProperty("--accent2", "#00C2D4");
    app.style.setProperty("--accent-glow", "#16E6A088");
    app.style.setProperty("--bad", "#FF5A6E");
  } else {
    app.style.setProperty("--accent", "#2d8f5f");
    app.style.setProperty("--accent2", "#5cb85c");
    app.style.setProperty("--accent-glow", "#2d8f5f44");
    app.style.setProperty("--bad", "#E11D48");
  }
  const btn = document.querySelector<HTMLButtonElement>('[data-role="btn-theme"]');
  if (!btn) return;
  const label = dark ? "Switch to light mode" : "Switch to dark mode";
  btn.title = label;
  btn.setAttribute("aria-label", label);
  btn.innerHTML = dark ? ICON_SUN : ICON_MOON;
}

function startClock(): void {
  const clock = document.querySelector<HTMLElement>('[data-role="shell-clock"]');
  if (!clock) return;
  const tick = () => {
    clock.textContent = new Date().toLocaleTimeString(undefined, { hour12: false });
  };
  tick();
  window.setInterval(tick, 1000);
}

export function mountShell(app: HTMLElement, siteLabel: string): void {
  const badge = document.querySelector<HTMLElement>('[data-role="site-badge"]');
  if (badge) badge.textContent = siteLabel;
  applyTheme(app, readDark());
  document.querySelector<HTMLButtonElement>('[data-role="btn-theme"]')?.addEventListener("click", () => {
    const nextDark = !app.classList.contains("theme-dark");
    applyTheme(app, nextDark);
    writeDark(nextDark);
  });
  startClock();
}
