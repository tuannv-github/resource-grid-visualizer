/** Strip `**` from OpenAPI-style emphasis in API prose. */
export function plainLegendNote(s: string): string {
  return s.replace(/\*\*/g, "");
}

export function axisTickValues(maxIdx: number, count: number): number[] {
  if (maxIdx < 0) return [0];
  const n = Math.min(count, maxIdx + 1);
  if (n <= 1) return [0];
  const vals: number[] = [];
  for (let i = 0; i < n; i++) {
    vals.push(Math.round((i * maxIdx) / (n - 1)));
  }
  return [...new Set(vals)].sort((a, b) => a - b);
}
