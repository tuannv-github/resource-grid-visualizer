/** Vite client env: treat common truthy strings as on. */
export function envTruthy(v: string | undefined): boolean {
  const t = v?.trim().toLowerCase();
  return t === "1" || t === "true" || t === "yes" || t === "on";
}
