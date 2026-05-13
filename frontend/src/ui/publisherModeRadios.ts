import {
  PUBLISHER_MODES,
  STREAM_MODE_SYNTHETIC_CHANNEL_BASED,
} from "../domain/streamModes";

export function mountPublisherModeRadios(
  fieldset: HTMLElement,
  streamRadioName: string,
): void {
  for (const { num, mode } of PUBLISHER_MODES) {
    const label = document.createElement("label");
    label.className = "mode-label";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = streamRadioName;
    input.value = mode;
    if (mode === STREAM_MODE_SYNTHETIC_CHANNEL_BASED) input.checked = true;
    const numSpan = document.createElement("span");
    numSpan.className = "mode-num";
    numSpan.textContent = String(num);
    const macro = document.createElement("code");
    macro.className = "mode-macro";
    macro.textContent = mode;
    label.append(input, numSpan, macro);
    fieldset.appendChild(label);
  }
}
