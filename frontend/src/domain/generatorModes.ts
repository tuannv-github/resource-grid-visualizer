/**
 * Process **GENERATION_MODE** / UI **mode** tokens (must match backend **GET /api/generator/status** `generation_mode`).
 * **POST /api/plotter/start** does not accept **`mode`** — use **POST /api/generator/start** with **`feed`** for producer selection.
 */
export const SYNTHETIC_IMAGES = "SYNTHETIC_IMAGES" as const;
export const SYNTHETIC_RESOURCE_GRID = "SYNTHETIC_RESOURCE_GRID" as const;
export const SYNTHETIC_FAPI = "SYNTHETIC_FAPI" as const;

export type GeneratorFeedStartMode =
  | typeof SYNTHETIC_IMAGES
  | typeof SYNTHETIC_RESOURCE_GRID
  | typeof SYNTHETIC_FAPI;

/** **`feed`** for **POST /api/generator/start** (short names; maps to synthetic modes server-side). */
export type GeneratorStartFeed = "images" | "resource_grid" | "fapi";

/** JSON for **POST /api/generator/start**. */
export type GeneratorStartBody =
  | { feed: "images"; images?: { pattern?: string; preset?: string } }
  | { feed: "resource_grid" }
  | { feed: "fapi" };
