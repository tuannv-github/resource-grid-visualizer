/*
 * Raster stage: CUDA kernels, CPU overlays, merge/acquire, render, `RasterStage::raster_run` → `RgbaFrameQueue`.
 */
#include "plotter_common.cuh"
#include "plotter_raster.cuh"
#include "plotter_posix_mq.cuh"
#include "viridis_lut.inc"
#include "font8x8_basic.inc"

#include <atomic>
#include <thread>
#include <vector>

/** True after a non-placeholder slot has been accepted. Later zero-slot placeholders are dropped. */
static std::atomic<bool> g_saw_live_slot{false};

static bool accept_acquired_frame(const GridFrame& gf) {
  if (gf.placeholder && g_saw_live_slot.load(std::memory_order_relaxed)) return false;
  if (!gf.placeholder) g_saw_live_slot.store(true, std::memory_order_relaxed);
  return true;
}

/** Implemented in `plotter_assembler.cu`; used here for standalone ``--mq-resource-grid`` (raster-owned MQ thread). */
void external_grid_frame_reader_main(ResgridMqStreamIn* mq, int ncells, GridFrameQueue* outq, const char* producer_stage);
void resource_grid_slot_iq_fifo_reader_main(ResgridMqStreamIn* mq, int prb_dim, int sym_dim, int ncells, GridFrameQueue* outq);

__device__ float3 viridis_rgb(float t) {
  t = fminf(fmaxf(t, 0.f), 1.f);
  const float x = t * 255.f;
  const int i = (int)x;
  const float f = x - (float)i;
  const int i1 = (i < 255) ? (i + 1) : 255;
  const float3 a = kViridis256[i];
  const float3 b = kViridis256[i1];
  float3 o;
  o.x = a.x + (b.x - a.x) * f;
  o.y = a.y + (b.y - a.y) * f;
  o.z = a.z + (b.z - a.z) * f;
  return o;
}

/** Base RGB for logical channel ids 1…8 (before optional RNTI modulation in `rgba_frame_kernel`). */
__device__ float3 channel_tint_rgb(uint8_t ch) {
  switch (ch) {
    case 1:
      return make_float3(0.92f, 0.25f, 0.28f); /* SSB */
    case 2:
      return make_float3(0.15f, 0.82f, 0.82f); /* PRACH */
    case 3:
      return make_float3(0.95f, 0.62f, 0.15f); /* PDCCH */
    case 4:
      return make_float3(0.78f, 0.38f, 0.92f); /* PUCCH */
    case 5:
      return make_float3(0.22f, 0.82f, 0.38f); /* PDSCH */
    case 6:
      return make_float3(0.28f, 0.45f, 0.95f); /* PUSCH */
    case 7:
      return make_float3(0.55f, 0.55f, 0.55f); /* CSI-RS */
    case 8:
      return make_float3(0.55f, 0.55f, 0.55f); /* SRS */
    default:
      return make_float3(0.55f, 0.55f, 0.55f);
  }
}

/** Subtle per-RNTI RGB jitter on top of `channel_tint_rgb` so e.g. two PDCCH/PUSCH UEs differ in tint. */
__device__ float3 channel_tint_modulated(uint8_t ch, uint32_t rnti) {
  float3 t = channel_tint_rgb(ch);
  if (rnti == 0u || ch < 1u || ch > 8u) return t;
  const float scale = 0.11f;
  const float a = (float)((rnti * 1664525u + 1013904223u) & 255u) * (1.f / 127.5f) - 1.f;
  const float b = (float)((rnti * 22695477u + 1u) & 255u) * (1.f / 127.5f) - 1.f;
  const float c = (float)((rnti * 1103515245u + 12345u) & 255u) * (1.f / 127.5f) - 1.f;
  t.x = fminf(fmaxf(t.x + a * scale * 0.7f, 0.f), 1.f);
  t.y = fminf(fmaxf(t.y + b * scale * 0.85f, 0.f), 1.f);
  t.z = fminf(fmaxf(t.z + c * scale * 0.75f, 0.f), 1.f);
  return t;
}

__global__ void __launch_bounds__(256) magnitude_kernel(const float2* z, float* mag, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mag[i] = hypotf(z[i].x, z[i].y);
}

/** Index-based upscaling: channel 0 → Viridis(mag); PHY ids 1…8 → flat tint from `cell_rnti` (RNTI jitter all PHY).
 *  `sym_start`/`sym_count` crop the data array along the symbol axis so only columns
 *  `[sym_start, sym_start+sym_count-1]` of the full `sym`-wide grid are rasterized. */
__global__ void rgba_frame_kernel(const float* mag, const uint8_t* chan, const uint32_t* cell_rnti, int prb, int sym,
                                  int sym_start, int sym_count, int frame_w, int frame_h, int off_x, int off_y,
                                  int content_w, int content_h, float mag_scale, uint8_t* rgba) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame_w || y >= frame_h) return;
  const int o = (y * frame_w + x) * 4;
  const int lx = x - off_x;
  const int ly = y - off_y;
  if (lx < 0 || ly < 0 || lx >= content_w || ly >= content_h) {
    rgba[o + 0] = rgba[o + 1] = rgba[o + 2] = 0;
    rgba[o + 3] = 255;
    return;
  }
  const int vs = min(sym_count - 1, (lx * sym_count) / max(1, content_w));
  const int s = min(sym - 1, sym_start + vs);
  /* PRB 0 at bottom of frame (screen y increases downward). */
  const int p = min(prb - 1, ((content_h - 1 - ly) * prb) / max(1, content_h));
  const int cell = s * prb + p;
  const uint8_t c = chan[cell];
  float3 rgb;
  if (c == 0) {
    const float m = mag[cell];
    const float t = fminf(fmaxf(m / mag_scale, 0.f), 1.f);
    rgb = viridis_rgb(t);
  } else {
    const uint32_t rn = cell_rnti[cell];
    const float3 tint = channel_tint_modulated(c, rn);
    rgb.x = fminf(tint.x, 1.f);
    rgb.y = fminf(tint.y, 1.f);
    rgb.z = fminf(tint.z, 1.f);
  }
  rgba[o + 0] = (uint8_t)fminf(rgb.x * 255.f, 255.f);
  rgba[o + 1] = (uint8_t)fminf(rgb.y * 255.f, 255.f);
  rgba[o + 2] = (uint8_t)fminf(rgb.z * 255.f, 255.f);
  rgba[o + 3] = 255;
}
static void rgba_blend_solid(uint8_t* rgba, int fw, int fh, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || y < 0 || x >= fw || y >= fh) return;
  uint8_t* p = rgba + ((size_t)y * (size_t)fw + (size_t)x) * 4u;
  p[0] = r;
  p[1] = g;
  p[2] = b;
  p[3] = 255;
}

static void rgba_blend_over(uint8_t* rgba, int fw, int fh, int x, int y, uint8_t r, uint8_t g, uint8_t b,
                            unsigned alpha) {
  if (x < 0 || y < 0 || x >= fw || y >= fh) return;
  uint8_t* p = rgba + ((size_t)y * (size_t)fw + (size_t)x) * 4u;
  const unsigned ia = 255u - alpha;
  p[0] = (uint8_t)((r * alpha + p[0] * ia + 127u) / 255u);
  p[1] = (uint8_t)((g * alpha + p[1] * ia + 127u) / 255u);
  p[2] = (uint8_t)((b * alpha + p[2] * ia + 127u) / 255u);
  p[3] = 255;
}

static void draw_glyph8(uint8_t* rgba, int fw, int fh, int x0, int y0, char c, int scale, uint8_t fr, uint8_t fg,
                        uint8_t fb) {
  auto uc = static_cast<unsigned char>(c);
  if (uc > 127) uc = static_cast<unsigned char>('?');
  for (int row = 0; row < 8; ++row) {
    const uint8_t bits = kFont8x8Basic[uc][row];
    /* font8x8 rows: LSB = leftmost pixel (https://github.com/dhepper/font8x8). MSB-first decoding mirrors glyphs. */
    for (int col = 0; col < 8; ++col) {
      if ((bits >> col) & 1) {
        for (int sy = 0; sy < scale; ++sy) {
          for (int sx = 0; sx < scale; ++sx) {
            const int px = x0 + col * scale + sx;
            const int py = y0 + row * scale + sy;
            rgba_blend_solid(rgba, fw, fh, px, py, fr, fg, fb);
          }
        }
      }
    }
  }
}

static void draw_text_line(uint8_t* rgba, int fw, int fh, int x0, int y0, const char* s, int scale, uint8_t fr,
                           uint8_t fg, uint8_t fb) {
  int x = x0;
  for (; *s; ++s) {
    draw_glyph8(rgba, fw, fh, x, y0, *s, scale, fr, fg, fb);
    x += 8 * scale + scale;
  }
}

/** PHY short name for channel id 1…8 (matches CUDA channel_tint_rgb). */
static void chan_phy_name(uint8_t ch_id, char* buf, size_t buflen) {
  switch (ch_id) {
    case 1:
      std::snprintf(buf, buflen, "SSB");
      break;
    case 2:
      std::snprintf(buf, buflen, "PRACH");
      break;
    case 3:
      std::snprintf(buf, buflen, "PDCCH");
      break;
    case 4:
      std::snprintf(buf, buflen, "PUCCH");
      break;
    case 5:
      std::snprintf(buf, buflen, "PDSCH");
      break;
    case 6:
      std::snprintf(buf, buflen, "PUSCH");
      break;
    case 7:
      std::snprintf(buf, buflen, "CSI-RS");
      break;
    case 8:
      std::snprintf(buf, buflen, "SRS");
      break;
    default:
      std::snprintf(buf, buflen, "%u", static_cast<unsigned>(ch_id));
      break;
  }
}

/** Inverse of `rgba_frame_kernel` cell indexing for the *visible* symbol column index `vs`
 *  (`vs == s - sym_start`); column `vs` spans `lx` in [`lx0`, `lx1`) within the cropped raster. */
static int resgrid_cell_lx0(int vs, int sym_count, int cw) {
  if (vs <= 0) return 0;
  return (vs * cw + sym_count - 1) / sym_count;
}

static int resgrid_cell_lx1_excl(int vs, int sym_count, int cw) {
  if (vs >= sym_count - 1) return cw;
  return ((vs + 1) * cw + sym_count - 1) / sym_count;
}

/** PRB `p` spans `ly` in [`ly0`, `ly1`) in content pixels (PRB 0 at bottom; small `ly` = top). */
static void resgrid_cell_ly_range(int p, int prb, int ch, int* ly0, int* ly1_excl) {
  const int u_lo = (p * ch + prb - 1) / prb;
  const int u_hi = ((p + 1) * ch + prb - 1) / prb;
  *ly0 = ch - u_hi;
  *ly1_excl = ch - u_lo;
}

/** True if `(s,p)` is not separated from `(ns,np)` by PHY type or RNTI (matches overlay flood-fill rules).
 *  Neighbors outside the visible symbol window `[sym_start, sym_start + sym_count)` are treated as a hard
 *  border so cropped edges always paint a stroke (matches what the rasterizer actually draws). */
static bool resgrid_phy_merges_with(const std::vector<uint8_t>& chan, const std::vector<uint32_t>& cell_rnti, int prb,
                                   int sym, int sym_start, int sym_count, int s, int p, int ns, int np,
                                   bool show_rnti) {
  const size_t ia = (size_t)s * (size_t)prb + (size_t)p;
  const uint8_t ca = chan[ia];
  if (ca < 1u || ca > 8u) return true;

  if (ns < sym_start || ns >= sym_start + sym_count || ns >= sym || np < 0 || np >= prb) return false;
  const size_t ib = (size_t)ns * (size_t)prb + (size_t)np;
  if (chan[ib] != ca) return false;
  if (show_rnti && cell_rnti[ia] != cell_rnti[ib]) return false;
  return true;
}

static void resgrid_blend_pixel(uint8_t* rgba, int fw, int fh, int x, int y, uint8_t r, uint8_t g, uint8_t b,
                                unsigned alpha) {
  if (x < 0 || y < 0 || x >= fw || y >= fh) return;
  rgba_blend_over(rgba, fw, fh, x, y, r, g, b, alpha);
}

/** Axis-aligned strokes in **frame** pixels; clips to the image. */
static void resgrid_frame_hline_stroke(uint8_t* rgba, int fw, int fh, int x0, int x1_excl, int y, int stroke,
                                       int dir_y, uint8_t r, uint8_t g, uint8_t b, unsigned alpha) {
  for (int t = 0; t < stroke; ++t) {
    const int yy = y + t * dir_y;
    if (yy < 0 || yy >= fh) continue;
    for (int x = x0; x < x1_excl; ++x) resgrid_blend_pixel(rgba, fw, fh, x, yy, r, g, b, alpha);
  }
}

static void resgrid_frame_vline_stroke(uint8_t* rgba, int fw, int fh, int x, int y0, int y1_excl, int stroke,
                                       int dir_x, uint8_t r, uint8_t g, uint8_t b, unsigned alpha) {
  for (int t = 0; t < stroke; ++t) {
    const int xx = x + t * dir_x;
    if (xx < 0 || xx >= fw) continue;
    for (int y = y0; y < y1_excl; ++y) resgrid_blend_pixel(rgba, fw, fh, xx, y, r, g, b, alpha);
  }
}

/** Draw 1-pixel borders between slot PHY regions (and vs magnitude background) using the same merge rule as labels.
 *  Iterates only the visible columns `[sym_start, sym_start + sym_count)` so cropping with
 *  `--start-symbol` / `--stop-symbol` matches the rasterized image. */
static void overlay_chan_region_borders_on_rgba(uint8_t* rgba, int fw, int fh, const std::vector<uint8_t>& chan,
                                                const std::vector<uint32_t>& cell_rnti, int prb, int sym,
                                                int sym_start, int sym_count, int off_x, int off_y, int content_w,
                                                int content_h) {
  if (chan.size() != (size_t)prb * (size_t)sym || prb < 1 || sym < 1 || sym_count < 1 || content_w < 1 ||
      content_h < 1)
    return;
  const bool show_rnti = cell_rnti.size() == chan.size();

  int stroke = content_w / (sym_count * 6);
  if (stroke < 1) stroke = 1;
  if (stroke > 3) stroke = 3;

  const unsigned alpha = 238u;
  const uint8_t br = 12u;
  const uint8_t bg = 12u;
  const uint8_t bb = 14u;

  for (int vs = 0; vs < sym_count; ++vs) {
    const int s = sym_start + vs;
    if (s < 0 || s >= sym) continue;
    const int lx0 = resgrid_cell_lx0(vs, sym_count, content_w);
    const int lx1 = resgrid_cell_lx1_excl(vs, sym_count, content_w);
    for (int p = 0; p < prb; ++p) {
      const size_t idx = (size_t)s * (size_t)prb + (size_t)p;
      if (chan[idx] < 1u || chan[idx] > 8u) continue;

      int ly0 = 0;
      int ly1 = 0;
      resgrid_cell_ly_range(p, prb, content_h, &ly0, &ly1);
      if (ly1 <= ly0 || lx1 <= lx0) continue;
      const int fx0 = off_x + lx0;
      const int fx1 = off_x + lx1;
      const int fy0 = off_y + ly0;
      const int fy1 = off_y + ly1;

      if (!resgrid_phy_merges_with(chan, cell_rnti, prb, sym, sym_start, sym_count, s, p, s - 1, p, show_rnti))
        resgrid_frame_vline_stroke(rgba, fw, fh, fx0, fy0, fy1, stroke, 1, br, bg, bb, alpha);
      if (!resgrid_phy_merges_with(chan, cell_rnti, prb, sym, sym_start, sym_count, s, p, s + 1, p, show_rnti))
        resgrid_frame_vline_stroke(rgba, fw, fh, fx1 - 1, fy0, fy1, stroke, -1, br, bg, bb, alpha);

      if (!resgrid_phy_merges_with(chan, cell_rnti, prb, sym, sym_start, sym_count, s, p, s, p + 1, show_rnti))
        resgrid_frame_hline_stroke(rgba, fw, fh, fx0, fx1, fy0, stroke, 1, br, bg, bb, alpha);
      if (!resgrid_phy_merges_with(chan, cell_rnti, prb, sym, sym_start, sym_count, s, p, s, p - 1, show_rnti))
        resgrid_frame_hline_stroke(rgba, fw, fh, fx0, fx1, fy1 - 1, stroke, -1, br, bg, bb, alpha);
    }
  }
}

/** PHY+RNTI label centered on (cx,cy) in frame pixels (4-connected region centroid in raster space). */
static void draw_centered_phy_rnti_label(uint8_t* rgba, int fw, int fh, int cx, int cy, const char* line1,
                                        const char* line2, bool show_rnti) {
  const int scale = 2;
  const int line_h = 8 * scale;
  const int adv = 8 * scale + scale;
  const int pad = 4;
  const int line_gap = 4;
  const int tw1 = static_cast<int>(std::strlen(line1)) * adv;
  const int tw2 = show_rnti ? static_cast<int>(std::strlen(line2)) * adv : 0;
  const int tw = std::max(tw1, tw2);
  const int inner_h = show_rnti ? (line_h + line_gap + line_h) : line_h;
  int x0 = cx - tw / 2 - pad;
  int y0 = cy - inner_h / 2 - pad;
  const int box_w = tw + 2 * pad;
  const int box_h = inner_h + 2 * pad;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x0 + box_w > fw) x0 = std::max(0, fw - box_w);
  if (y0 + box_h > fh) y0 = std::max(0, fh - box_h);
  for (int yy = y0; yy < y0 + box_h && yy < fh; ++yy) {
    for (int xx = x0; xx < x0 + box_w && xx < fw; ++xx) {
      rgba_blend_over(rgba, fw, fh, xx, yy, 8, 8, 16, 200);
    }
  }
  const int tx1 = x0 + pad + (tw - tw1) / 2;
  draw_text_line(rgba, fw, fh, tx1, y0 + pad, line1, scale, 252, 252, 245);
  if (show_rnti) {
    const int tx2 = x0 + pad + (tw - tw2) / 2;
    draw_text_line(rgba, fw, fh, tx2, y0 + pad + line_h + line_gap, line2, scale, 230, 240, 255);
  }
}

/** One label per 4-connected region of cells chan 1…8. When cell_rnti is populated, neighbors must also
 *  match RNTI so two regions with the same PHY id but different RNTIs (any of PDCCH/PDSCH/PUCCH/PUSCH/SRS/…) get
 *  separate centroid labels; CUDA uses the same per-cell RNTI for a subtle tint jitter on ids 1…8.
 *  Flood fill is clipped to the visible symbol window `[sym_start, sym_start + sym_count)` so labels and
 *  centroids reflect only the cropped raster. */
static void overlay_chan_region_labels_on_rgba(uint8_t* rgba, int fw, int fh, const std::vector<uint8_t>& chan,
                                              const std::vector<uint32_t>& cell_rnti, int prb, int sym, int sym_start,
                                              int sym_count, int off_x, int off_y, int content_w, int content_h) {
  if (chan.size() != (size_t)prb * (size_t)sym || prb < 1 || sym < 1 || sym_count < 1 || content_w < 1 ||
      content_h < 1)
    return;
  const bool show_rnti = cell_rnti.size() == chan.size();
  const size_t n = (size_t)prb * (size_t)sym;
  std::vector<uint8_t> vis(n, 0);
  static const int d4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  const int sym_stop_excl = std::min(sym, sym_start + sym_count);

  for (int s = sym_start; s < sym_stop_excl; ++s) {
    for (int p = 0; p < prb; ++p) {
      const size_t idx = (size_t)s * (size_t)prb + (size_t)p;
      const uint8_t c0 = chan[idx];
      if (c0 < 1u || c0 > 8u || vis[idx]) continue;

      const uint32_t r0 = show_rnti ? cell_rnti[idx] : 0u;

      double sum_lx = 0.0;
      double sum_ly = 0.0;
      size_t ngeom = 0;
      double sum_vs = 0.0;
      double sum_p = 0.0;
      size_t cnt = 0;
      std::vector<size_t> st;
      st.push_back(idx);
      vis[idx] = 1;

      while (!st.empty()) {
        const size_t u = st.back();
        st.pop_back();
        const int us = (int)(u / (size_t)prb);
        const int up = (int)(u % (size_t)prb);
        const int uvs = us - sym_start;
        sum_vs += (double)uvs;
        sum_p += (double)up;
        ++cnt;

        const int lx0 = resgrid_cell_lx0(uvs, sym_count, content_w);
        const int lx1 = resgrid_cell_lx1_excl(uvs, sym_count, content_w);
        int cly0 = 0;
        int cly1 = 0;
        resgrid_cell_ly_range(up, prb, content_h, &cly0, &cly1);
        if (lx1 > lx0 && cly1 > cly0) {
          sum_lx += 0.5 * (double)(lx0 + lx1);
          sum_ly += 0.5 * (double)(cly0 + cly1);
          ++ngeom;
        }

        for (int di = 0; di < 4; ++di) {
          const int ns = us + d4[di][0];
          const int np = up + d4[di][1];
          if (ns < sym_start || ns >= sym_stop_excl || np < 0 || np >= prb) continue;
          const size_t v = (size_t)ns * (size_t)prb + (size_t)np;
          if (vis[v] || chan[v] != c0) continue;
          if (show_rnti && cell_rnti[v] != r0) continue;
          vis[v] = 1;
          st.push_back(v);
        }
      }

      float lx;
      float ly;
      if (ngeom > 0) {
        lx = (float)(sum_lx / (double)ngeom);
        ly = (float)(sum_ly / (double)ngeom);
      } else if (cnt > 0) {
        const float mvs = (float)(sum_vs / (double)cnt);
        const float mp = (float)(sum_p / (double)cnt);
        lx = (mvs + 0.5f) * (float)content_w / (float)sym_count;
        ly = (float)content_h - (mp + 0.5f) * (float)content_h / (float)prb;
      } else {
        continue;
      }

      const int pcx = off_x + (int)std::lrintf(lx);
      const int pcy = off_y + (int)std::lrintf(ly);

      char line1[16];
      chan_phy_name(static_cast<uint8_t>(c0), line1, sizeof line1);
      char line2[40];
      if (show_rnti)
        std::snprintf(line2, sizeof line2, "0x%X", static_cast<unsigned>(r0));
      else
        line2[0] = '\0';

      draw_centered_phy_rnti_label(rgba, fw, fh, pcx, pcy, line1, line2, show_rnti);
    }
  }
}
bool pop_merged_assembled_or_sidecar_iq(AssembledSlotQueue* assembled, GridFrameQueue* iq_q, GridFrame& gf,
                                               SlotChanRntiTemplate* last_slot_meta, int ncells) {
  auto capture_meta = [&] {
    if (!last_slot_meta) return;
    last_slot_meta->chan = gf.chan;
    last_slot_meta->cell_rnti = gf.cell_rnti;
  };
  auto merge_sidecar_meta = [&] {
    if (!last_slot_meta || (int)last_slot_meta->chan.size() != ncells) return;
    gf.chan = last_slot_meta->chan;
    gf.cell_rnti = last_slot_meta->cell_rnti;
  };

  if (assembled->try_pop(gf) && accept_acquired_frame(gf)) {
    capture_meta();
    return true;
  }
  if (iq_q && iq_q->try_pop(gf) && accept_acquired_frame(gf)) {
    merge_sidecar_meta();
    return true;
  }
  if (iq_q) {
    if (assembled->pop_wait(gf, 80) && accept_acquired_frame(gf)) {
      capture_meta();
      return true;
    }
    if (iq_q->pop(gf, 80) && accept_acquired_frame(gf)) {
      merge_sidecar_meta();
      return true;
    }
    return false;
  }
  if (assembled->pop_wait(gf, 400) && accept_acquired_frame(gf)) {
    capture_meta();
    return true;
  }
  return false;
}

bool acquire_frame_for_interpolate(InternalFrameQueue* internal_q, FullGridFifoQueue* full_grid_fifo_q,
                                          AssembledSlotQueue* slot_assembled_q, GridFrameQueue* iq_sidecar_q,
                                          GridFrame& gf, SlotChanRntiTemplate* last_slot_meta, int ncells) {
  if (slot_assembled_q)
    return pop_merged_assembled_or_sidecar_iq(slot_assembled_q, iq_sidecar_q, gf, last_slot_meta, ncells);
  if (full_grid_fifo_q) return full_grid_fifo_q->pop(gf, 2000);
  if (internal_q) return internal_q->pop(gf, 2000);
  return false;
}

/** Stroke ``boxes`` from the resource-grid header. PRB 0 is the bottom row; symbols run left to right. */
static void overlay_grid_boxes_on_rgba(uint8_t* rgba, int fw, int fh, const std::vector<GridBox>& boxes, int prb, int sym,
                                       int sym_start, int sym_count, int off_x, int off_y, int content_w, int content_h) {
  if (boxes.empty() || prb < 1 || sym < 1 || sym_count < 1 || content_w < 1 || content_h < 1) return;
  int stroke = content_w / (sym_count * 4);
  if (stroke < 2) stroke = 2;
  if (stroke > 4) stroke = 4;
  const uint8_t br = 255;
  const uint8_t bg = 196;
  const uint8_t bb = 64;
  const unsigned alpha = 235u;
  for (const GridBox& box : boxes) {
    if (box.prb_count == 0 || box.sym_count == 0) continue;
    int p0 = static_cast<int>(box.prb_start);
    int p1 = p0 + static_cast<int>(box.prb_count);
    if (p0 < 0) p0 = 0;
    if (p1 > prb) p1 = prb;
    if (p0 >= p1) continue;
    int s0 = std::max(static_cast<int>(box.sym_start), sym_start);
    int s1 = std::min(static_cast<int>(box.sym_start) + static_cast<int>(box.sym_count), sym_start + sym_count);
    s0 = std::max(s0, 0);
    s1 = std::min(s1, sym);
    if (s0 >= s1) continue;
    const int vs0 = s0 - sym_start;
    const int vs1 = s1 - sym_start;
    const int lx0 = resgrid_cell_lx0(vs0, sym_count, content_w);
    const int lx1 = resgrid_cell_lx1_excl(vs1 - 1, sym_count, content_w);
    int top0 = 0;
    int top1 = 0;
    int bot0 = 0;
    int bot1 = 0;
    resgrid_cell_ly_range(p1 - 1, prb, content_h, &top0, &top1);
    resgrid_cell_ly_range(p0, prb, content_h, &bot0, &bot1);
    const int fx0 = off_x + lx0;
    const int fx1 = off_x + lx1;
    const int fy0 = off_y + top0;
    const int fy1 = off_y + bot1;
    if (fx1 <= fx0 || fy1 <= fy0) continue;
    resgrid_frame_hline_stroke(rgba, fw, fh, fx0, fx1, fy0, stroke, 1, br, bg, bb, alpha);
    resgrid_frame_hline_stroke(rgba, fw, fh, fx0, fx1, fy1 - 1, stroke, -1, br, bg, bb, alpha);
    resgrid_frame_vline_stroke(rgba, fw, fh, fx0, fy0, fy1, stroke, 1, br, bg, bb, alpha);
    resgrid_frame_vline_stroke(rgba, fw, fh, fx1 - 1, fy0, fy1, stroke, -1, br, bg, bb, alpha);
  }
}

/** CUDA interpolate (mag + tint) → host RGBA + CPU overlays (encoder input buffer).
 *  `sym_start` / `sym_count` crop the raster to the visible symbol window. */
void render_frame_rgba(const GridFrame& gf, float2* d_z, float* d_mag, uint8_t* d_chan, uint32_t* d_cell_rnti,
                              uint8_t* d_rgba, std::vector<uint8_t>& h_rgba, int ncells, int prb, int sym,
                              int sym_start, int sym_count, int frame_w, int frame_h, int off_x, int off_y,
                              int content_w, int content_h, float mag_scale, dim3 grid_mag, dim3 block_mag,
                              dim3 grid2, dim3 block2, cudaStream_t stream) {
  const cudaStream_t st = stream ? stream : (cudaStream_t)0;
  cuda_check(cudaMemcpyAsync(d_z, gf.iq.data(), ncells * sizeof(float2), cudaMemcpyHostToDevice, st), "H2D z");
  cuda_check(cudaMemcpyAsync(d_chan, gf.chan.data(), (size_t)ncells * sizeof(uint8_t), cudaMemcpyHostToDevice, st),
             "H2D chan");
  if ((int)gf.cell_rnti.size() == ncells)
    cuda_check(cudaMemcpyAsync(d_cell_rnti, gf.cell_rnti.data(), (size_t)ncells * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, st),
               "H2D cell_rnti");
  else
    cuda_check(cudaMemsetAsync(d_cell_rnti, 0, (size_t)ncells * sizeof(uint32_t), st), "memset cell_rnti");

  magnitude_kernel<<<grid_mag, block_mag, 0, st>>>(d_z, d_mag, ncells);
  cuda_check(cudaGetLastError(), "magnitude_kernel");
  rgba_frame_kernel<<<grid2, block2, 0, st>>>(d_mag, d_chan, d_cell_rnti, prb, sym, sym_start, sym_count, frame_w,
                                              frame_h, off_x, off_y, content_w, content_h, mag_scale, d_rgba);
  cuda_check(cudaGetLastError(), "rgba_frame_kernel");
  cuda_check(cudaMemcpyAsync(h_rgba.data(), d_rgba, h_rgba.size(), cudaMemcpyDeviceToHost, st), "D2H rgba");
  cuda_check(cudaStreamSynchronize(st), "raster cuda stream");

  bool any_chan_region = false;
  for (uint8_t v : gf.chan) {
    if (v >= 1u && v <= 8u) {
      any_chan_region = true;
      break;
    }
  }
  if (any_chan_region) {
    overlay_chan_region_borders_on_rgba(h_rgba.data(), frame_w, frame_h, gf.chan, gf.cell_rnti, prb, sym, sym_start,
                                       sym_count, off_x, off_y, content_w, content_h);
    overlay_chan_region_labels_on_rgba(h_rgba.data(), frame_w, frame_h, gf.chan, gf.cell_rnti, prb, sym, sym_start,
                                       sym_count, off_x, off_y, content_w, content_h);
  }
  if (gf.has_timing) {
    char timing[48];
    std::snprintf(timing, sizeof(timing), "SFN %u SLOT %u", static_cast<unsigned>(gf.sfn), static_cast<unsigned>(gf.slot));
    const int scale = 2;
    const int adv = 8 * scale + scale;
    const int tw = static_cast<int>(std::strlen(timing)) * adv;
    const int pad = 4;
    const int box_w = tw + 2 * pad;
    const int box_h = 8 * scale + 2 * pad;
    const int x0 = 8;
    const int y0 = 8;
    for (int yy = y0; yy < y0 + box_h && yy < frame_h; ++yy) {
      for (int xx = x0; xx < x0 + box_w && xx < frame_w; ++xx)
        rgba_blend_over(h_rgba.data(), frame_w, frame_h, xx, yy, 8, 8, 16, 200);
    }
    draw_text_line(h_rgba.data(), frame_w, frame_h, x0 + pad, y0 + pad, timing, scale, 252, 252, 245);
  }
  if (!gf.boxes.empty())
    overlay_grid_boxes_on_rgba(h_rgba.data(), frame_w, frame_h, gf.boxes, prb, sym, sym_start, sym_count, off_x, off_y,
                               content_w, content_h);
}

void fill_synthetic_iq(std::vector<float2>& out, std::vector<uint8_t>& chan, std::vector<uint32_t>& cell_rnti,
                              int prb, int sym, uint64_t frame, uint32_t pattern) {
  const size_t n = (size_t)prb * (size_t)sym;
  out.resize(n);
  chan.assign(n, 0u);
  cell_rnti.assign(n, 0u);
  /* Per-cell channel ids 1…8 (PHY tints + RNTI labels) come only from completed slot FIFO frames — not from SHM pattern. */
  if (pattern > 3u) pattern = 0u;

  for (int s = 0; s < sym; ++s) {
    for (int p = 0; p < prb; ++p) {
      const size_t idx = (size_t)s * (size_t)prb + (size_t)p;
      float re = 0.f, im = 0.f;
      if (pattern == 1u) {
        const float ph = 0.03f * (float)frame;
        re = 0.9f * cosf(ph);
        im = 0.9f * sinf(ph);
      } else if (pattern == 2u) {
        const float cx = (float)p / (float)prb - 0.5f;
        const float cy = (float)s / (float)sym - 0.5f;
        const float g = expf(-((cx * cx + cy * cy) / (2.f * 0.12f * 0.12f)));
        const float amp = 0.95f * g;
        const float ph = 0.02f * (float)frame;
        re = amp * cosf(ph);
        im = amp * sinf(ph);
      } else if (pattern == 3u) {
        const float cx = (float)p / (float)prb - 0.5f;
        const float cy = (float)s / (float)sym - 0.5f;
        const float r2 = cx * cx + cy * cy;
        const float g = expf(-r2 / (2.f * 0.35f * 0.35f));
        const float amp = 0.85f * g;
        const float th = std::atan2(cy, cx);
        const float ph = 2.f * th + 0.04f * (float)frame;
        re = amp * cosf(ph);
        im = amp * sinf(ph);
      } else {
        const float cx = (float)p / (float)prb - 0.5f;
        const float cy = (float)s / (float)sym - 0.5f;
        const float ph = 0.03f * (float)frame + 6.2831853f * (cx * 1.3f + cy * 0.9f);
        const float amp = 0.45f + 0.45f * sinf(ph * 0.7f + cx * 4.f);
        re = amp * cosf(ph);
        im = amp * sinf(ph);
      }
      out[idx] = make_float2(re, im);
    }
  }
}

void RasterStage::raster_configure(int rt, size_t rgba_bytes, InternalFrameQueue* internal_q,
                                   FullGridFifoQueue* full_grid_q, AssembledSlotQueue* assembled_q,
                                   GridFrameQueue* sidecar_q, bool resource_grid_slot_stream, RgbaFrameQueue* rgba_out,
                                   void* fps_base, int ncells, int prb, int sym, int sym_start, int sym_count,
                                   int frame_w, int frame_h, int off_x, int off_y, int content_w, int content_h,
                                   float mag_scale, dim3 grid_mag, dim3 block_mag, dim3 grid2, dim3 block2,
                                   CapturerStage* capturer) {
  rt_ = rt;
  rgba_bytes_ = rgba_bytes;
  merger_rgba_out_ = rgba_out;
  merger_capturer_ = capturer;

  acq_internal_q_ = internal_q;
  acq_full_grid_q_ = full_grid_q;
  acq_assembled_q_ = assembled_q;
  acq_sidecar_q_ = sidecar_q;
  acq_resource_grid_slot_stream_ = resource_grid_slot_stream;
  acq_ncells_ = ncells;
  acq_prb_ = prb;
  acq_sym_ = sym;

  threads_.resize((size_t)rt);
  for (int i = 0; i < rt; ++i) {
    PerThread& rc = threads_[(size_t)i];
    rc.internal_q = internal_q;
    rc.full_grid_q = full_grid_q;
    rc.assembled_q = assembled_q;
    rc.sidecar_q = sidecar_q;
    rc.resource_grid_slot_stream = resource_grid_slot_stream;
    rc.rgba_out = rgba_out;
    rc.fps_base = fps_base;
    rc.ncells = ncells;
    rc.prb = prb;
    rc.sym = sym;
    rc.sym_start = sym_start;
    rc.sym_count = sym_count;
    rc.frame_w = frame_w;
    rc.frame_h = frame_h;
    rc.off_x = off_x;
    rc.off_y = off_y;
    rc.content_w = content_w;
    rc.content_h = content_h;
    rc.mag_scale = mag_scale;
    rc.grid_mag = grid_mag;
    rc.block_mag = block_mag;
    rc.grid2 = grid2;
    rc.block2 = block2;
    rc.capturer = capturer;
  }
}

void RasterStage::raster_set_thread_cuda(int index, float2* d_z, float* d_mag, uint8_t* d_chan, uint32_t* d_cell_rnti,
                                  uint8_t* d_rgba, cudaStream_t cuda_stream) {
  PerThread& rc = threads_[(size_t)index];
  rc.d_z = d_z;
  rc.d_mag = d_mag;
  rc.d_chan = d_chan;
  rc.d_cell_rnti = d_cell_rnti;
  rc.d_rgba = d_rgba;
  rc.cuda_stream = cuda_stream;
}

void RasterStage::raster_configure_pool(RasterWorkQueue* work_q) {
  work_q_ = work_q;
  for (auto& rc : threads_) {
    rc.internal_q = nullptr;
    rc.full_grid_q = nullptr;
    rc.assembled_q = nullptr;
    rc.sidecar_q = nullptr;
    rc.rgba_out = nullptr;
    rc.capturer = nullptr;
  }
}

void RasterStage::raster_attach_full_grid_mq_reader(ResgridMqStreamIn* iq_mq, bool resource_grid_slot_iq_stream) {
  full_grid_mq_reader_ = iq_mq;
  full_grid_mq_reader_slot_stream_ = resource_grid_slot_iq_stream;
}

void RasterStage::single_thread_main() {
  PerThread& ctx = threads_[0];
  SlotChanRntiTemplate slot_meta;
  SlotChanRntiTemplate* slot_meta_ptr =
      (ctx.assembled_q && ctx.sidecar_q) ? &slot_meta : nullptr;
  const size_t rgba_bytes = (size_t)ctx.frame_w * (size_t)ctx.frame_h * 4u;
  std::vector<uint8_t> packet;
  packet.resize(rgba_bytes);
  bool produced_any = false;
  fprintf(stderr, "resgrid_plotter: Raster stage (Assembler→RGBA) thread started\n");
  while (g_run.load()) {
    GridFrame gf;
    const bool got = acquire_frame_for_interpolate(ctx.internal_q, ctx.full_grid_q, ctx.assembled_q, ctx.sidecar_q, gf,
                                                   slot_meta_ptr, ctx.ncells);
    bool did_render = false;
    if (got) {
      render_frame_rgba(gf, ctx.d_z, ctx.d_mag, ctx.d_chan, ctx.d_cell_rnti, ctx.d_rgba, packet, ctx.ncells, ctx.prb,
                        ctx.sym, ctx.sym_start, ctx.sym_count, ctx.frame_w, ctx.frame_h, ctx.off_x, ctx.off_y,
                        ctx.content_w, ctx.content_h, ctx.mag_scale, ctx.grid_mag, ctx.block_mag, ctx.grid2,
                        ctx.block2, ctx.cuda_stream);
      g_stage_output_count_raster.fetch_add(1, std::memory_order_relaxed);
      produced_any = true;
      did_render = true;
    }
    if (did_render) {
      if (ctx.capturer) ctx.capturer->maybe_write(packet.data(), rgba_bytes);
      ctx.rgba_out->push(std::move(packet), "Raster");
      packet.resize(rgba_bytes);
    } else {
      if ((ctx.assembled_q || ctx.resource_grid_slot_stream) && produced_any)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  fprintf(stderr, "resgrid_plotter: Raster stage thread exiting\n");
}

RasterWorkQueue::RasterWorkQueue(int capacity) : cap_(std::max(1, capacity)) {}

void RasterWorkQueue::push(RasterWorkItem&& frame, const char* /*producer_stage*/) {
  /* Each item already has a merger sequence number. Dropping one leaves a hole:
   * the merger waits for that seq and retains every later RGBA frame, which grows
   * without bound and OOM-kills the process. Block instead. The input queues
   * ahead of the acquirer are the ones that drop the oldest slot. */
  std::unique_lock<std::mutex> lk(mu_);
  not_full_.wait(lk, [&] {
    return (int)q_.size() < cap_ || !g_run.load(std::memory_order_relaxed);
  });
  if (!g_run.load(std::memory_order_relaxed)) return;
  q_.push_back(std::move(frame));
  not_empty_.notify_one();
}

bool RasterWorkQueue::pop(RasterWorkItem& out, int wait_ms) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!not_empty_.wait_for(lk, std::chrono::milliseconds(wait_ms), [&] {
        return !q_.empty() || !g_run.load(std::memory_order_relaxed);
      }))
    return false;
  if (q_.empty()) return false;
  out = std::move(q_.front());
  q_.pop_front();
  not_full_.notify_one();
  return true;
}

bool RasterWorkQueue::try_pop(RasterWorkItem& out) {
  std::lock_guard<std::mutex> lk(mu_);
  if (q_.empty()) return false;
  out = std::move(q_.front());
  q_.pop_front();
  not_full_.notify_one();
  return true;
}

void RasterStage::pool_submit_rgba(uint64_t seq, std::vector<uint8_t>&& rgba) {
  std::lock_guard<std::mutex> lk(pool_mu_);
  pool_pending_.emplace(seq, std::move(rgba));
  pool_cv_.notify_one();
}

void RasterStage::acquirer_main() {
  SlotChanRntiTemplate slot_meta;
  SlotChanRntiTemplate* slot_meta_ptr =
      (acq_assembled_q_ && acq_sidecar_q_) ? &slot_meta : nullptr;
  uint64_t issue_seq = 0;
  fprintf(stderr, "resgrid_plotter: Raster acquirer thread started (pool mode)\n");
  while (g_run.load(std::memory_order_relaxed)) {
    GridFrame gf;
    const bool got = acquire_frame_for_interpolate(acq_internal_q_, acq_full_grid_q_, acq_assembled_q_, acq_sidecar_q_,
                                                   gf, slot_meta_ptr, acq_ncells_);
    if (got) {
      RasterWorkItem wi;
      wi.seq = issue_seq++;
      wi.gf = std::move(gf);
      work_q_->push(std::move(wi), "Raster (acquirer)");
    } else if (acq_assembled_q_ || acq_resource_grid_slot_stream_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  {
    std::lock_guard<std::mutex> lk(pool_mu_);
    pool_acquire_issued_end_ = issue_seq;
    pool_acquire_done_ = true;
  }
  pool_cv_.notify_all();
  fprintf(stderr, "resgrid_plotter: Raster acquirer thread exiting\n");
}

void RasterStage::worker_main(int thread_index) {
  cuda_check(cudaSetDevice(0), "raster worker cudaSetDevice");
  PerThread& ctx = threads_[(size_t)thread_index];
  std::vector<uint8_t> packet;
  packet.resize(rgba_bytes_);
  while (g_run.load(std::memory_order_relaxed)) {
    RasterWorkItem item;
    if (!work_q_->pop(item, 200)) {
      if (!g_run.load(std::memory_order_relaxed)) break;
      continue;
    }
    render_frame_rgba(item.gf, ctx.d_z, ctx.d_mag, ctx.d_chan, ctx.d_cell_rnti, ctx.d_rgba, packet, ctx.ncells, ctx.prb,
                      ctx.sym, ctx.sym_start, ctx.sym_count, ctx.frame_w, ctx.frame_h, ctx.off_x, ctx.off_y,
                      ctx.content_w, ctx.content_h, ctx.mag_scale, ctx.grid_mag, ctx.block_mag, ctx.grid2, ctx.block2,
                      ctx.cuda_stream);
    pool_submit_rgba(item.seq, std::move(packet));
    packet.resize(rgba_bytes_);
  }
  RasterWorkItem drain;
  while (work_q_->try_pop(drain)) {
    render_frame_rgba(drain.gf, ctx.d_z, ctx.d_mag, ctx.d_chan, ctx.d_cell_rnti, ctx.d_rgba, packet, ctx.ncells, ctx.prb,
                      ctx.sym, ctx.sym_start, ctx.sym_count, ctx.frame_w, ctx.frame_h, ctx.off_x, ctx.off_y,
                      ctx.content_w, ctx.content_h, ctx.mag_scale, ctx.grid_mag, ctx.block_mag, ctx.grid2, ctx.block2,
                      ctx.cuda_stream);
    pool_submit_rgba(drain.seq, std::move(packet));
    packet.resize(rgba_bytes_);
  }
}

void RasterStage::merger_main() {
  fprintf(stderr, "resgrid_plotter: Raster merger thread started\n");
  for (;;) {
    std::unique_lock<std::mutex> lk(pool_mu_);
    pool_cv_.wait(lk, [&] {
      return pool_pending_.count(pool_next_merged_seq_) ||
             (pool_acquire_done_ && pool_next_merged_seq_ == pool_acquire_issued_end_) ||
             pool_pending_.size() > 64;
    });
    if (!pool_pending_.count(pool_next_merged_seq_) && pool_pending_.size() > 64) {
      uint64_t next = 0;
      bool found = false;
      for (const auto& kv : pool_pending_) {
        if (kv.first < pool_next_merged_seq_) continue;
        if (!found || kv.first < next) {
          next = kv.first;
          found = true;
        }
      }
      if (found && next > pool_next_merged_seq_) {
        fprintf(stderr,
                "resgrid_plotter: raster merger skipped %llu missing frame(s); %zu RGBA frames were buffered\n",
                (unsigned long long)(next - pool_next_merged_seq_), pool_pending_.size());
        pool_next_merged_seq_ = next;
      }
    }
    if (pool_pending_.count(pool_next_merged_seq_)) {
      auto it = pool_pending_.find(pool_next_merged_seq_);
      std::vector<uint8_t> pix = std::move(it->second);
      pool_pending_.erase(it);
      pool_next_merged_seq_++;
      lk.unlock();
      if (merger_capturer_) merger_capturer_->maybe_write(pix.data(), pix.size());
      merger_rgba_out_->push(std::move(pix), "Raster (merger)");
      g_stage_output_count_raster.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (pool_acquire_done_ && pool_next_merged_seq_ == pool_acquire_issued_end_) break;
  }
  fprintf(stderr, "resgrid_plotter: Raster merger thread exiting\n");
}

void RasterStage::raster_run() {
  fprintf(stderr, "resgrid_plotter: Raster stage thread started\n");
  std::thread full_grid_mq_reader;
  const bool run_full_grid_mq_reader =
      full_grid_mq_reader_ != nullptr && full_grid_mq_reader_->ok() && acq_full_grid_q_ != nullptr;
  if (run_full_grid_mq_reader) {
    ResgridMqStreamIn* const mq = full_grid_mq_reader_;
    FullGridFifoQueue* const fq = acq_full_grid_q_;
    const bool slot = full_grid_mq_reader_slot_stream_;
    const int prb_dim = acq_prb_, sym_dim = acq_sym_, ncells = acq_ncells_;
    full_grid_mq_reader = std::thread([mq, fq, slot, prb_dim, sym_dim, ncells]() {
      if (slot)
        resource_grid_slot_iq_fifo_reader_main(mq, prb_dim, sym_dim, ncells, fq);
      else
        external_grid_frame_reader_main(mq, ncells, fq, "Raster (IQ MQ full-grid)");
    });
  }

  if (rt_ == 1) {
    single_thread_main();
  } else {
    std::thread merger([this]() { merger_main(); });
    std::thread acquirer([this]() { acquirer_main(); });
    std::vector<std::thread> workers;
    for (int i = 0; i < rt_; ++i)
      workers.emplace_back([this, i]() { worker_main(i); });
    fprintf(stderr, "resgrid_plotter: Raster pool started (1 acquirer + %d CUDA workers + 1 merger)\n", rt_);
    acquirer.join();
    for (auto& w : workers) w.join();
    merger.join();
  }

  if (full_grid_mq_reader.joinable()) full_grid_mq_reader.join();
  fprintf(stderr, "resgrid_plotter: Raster stage thread exiting\n");
}
