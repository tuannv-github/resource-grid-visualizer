import { PUBLISHER_MODE_PRESETS, SYNTHETIC_FAPI } from "../domain/publisherModes";

/** Builds the **Mode** configuration row (same **`iq-cfg` / `iq-cfg-label`** layout as Pattern). */
export function mountModeSelect(mountRoot: HTMLElement, selectId: string): HTMLSelectElement {
  mountRoot.replaceChildren();
  const label = document.createElement("label");
  label.className = "iq-cfg-label";
  const sel = document.createElement("select");
  sel.id = selectId;
  sel.dataset.role = "mode-select";
  sel.className = "iq-cfg-input mode-select-input";
  sel.setAttribute("aria-label", "Mode");
  for (const { num, mode } of PUBLISHER_MODE_PRESETS) {
    const o = document.createElement("option");
    o.value = mode;
    o.textContent = mode;
    o.title = `${num}: ${mode}`;
    sel.appendChild(o);
  }
  sel.value = SYNTHETIC_FAPI;
  label.append("Mode ", sel);
  mountRoot.appendChild(label);
  return sel;
}
