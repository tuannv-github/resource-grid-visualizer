/** Query a descendant by `data-role` within a panel root. */
export function queryRole<T extends HTMLElement>(root: Element, role: string): T | null {
  return root.querySelector<T>(`[data-role="${role}"]`);
}

export function requireRole<T extends HTMLElement>(root: Element, role: string): T {
  const el = queryRole<T>(root, role);
  if (!el) throw new Error(`Missing [data-role="${role}"]`);
  return el;
}
