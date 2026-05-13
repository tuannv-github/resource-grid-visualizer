/*
 * Assembly stage: slot FIFO → `AssembledSlotQueue`; IQ FIFO readers; internal synthetic producer.
 */
#include "plotter_common.cuh"

#include <errno.h>
#include <fcntl.h>
#include <memory>
#include <poll.h>
#include <thread>
#include <unistd.h>
static constexpr uint32_t kSlotFifoMagic = 0x31564153u;
static constexpr uint16_t kSlotProtoVersion1 = 1u;
static constexpr uint16_t kSlotProtoVersion2 = 2u;
/** `SlotMsgHdr::flags`: compound record with `n_regions` descriptors after the 40-byte header. */
static constexpr uint16_t kSlotFlagMultiRegion = 1u;
static constexpr uint16_t kSlotMaxRegions = 512u;

static SlotRegionDesc slot_region_from_hdr(const SlotMsgHdr& h) {
  SlotRegionDesc r{};
  r.sym_start = h.sym_start;
  r.sym_count = h.sym_count;
  r.prb_start = h.prb_start;
  r.prb_count = h.prb_count;
  r.rnti = h.rnti;
  r.channel = h.channel;
  r.reserved[0] = r.reserved[1] = r.reserved[2] = 0;
  return r;
}
static bool read_exact_from(int fd, void* buf, size_t n, const char* ctx) {
  auto* p = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n && g_run.load()) {
    ssize_t r = read(fd, p + got, n - got);
    if (r < 0) {
      fprintf(stderr, "resgrid_plotter read %s: ", ctx);
      perror(nullptr);
      return false;
    }
    if (r == 0) return false;
    got += static_cast<size_t>(r);
  }
  return got == n && g_run.load();
}

static void scatter_slot_payload_patch(std::vector<float2>& grid, int prb_dim, int sym_dim, const SlotRegionDesc& r,
                                       const float2* payload, size_t payload_elems) {
  const uint64_t expect = (uint64_t)r.sym_count * (uint64_t)r.prb_count;
  if (expect == 0u || expect != (uint64_t)payload_elems) return;
  if ((int)r.sym_start + (int)r.sym_count > sym_dim || (int)r.prb_start + (int)r.prb_count > prb_dim) return;
  for (int s = 0; s < (int)r.sym_count; ++s) {
    for (int p = 0; p < (int)r.prb_count; ++p) {
      const int gs = (int)r.sym_start + s;
      const int gp = (int)r.prb_start + p;
      const size_t gidx = (size_t)gs * (size_t)prb_dim + (size_t)gp;
      const size_t pidx = (size_t)s * (size_t)r.prb_count + (size_t)p;
      grid[gidx] = payload[pidx];
    }
  }
}

static void scatter_slot_chan_patch(std::vector<uint8_t>& chan, int prb_dim, int sym_dim, const SlotRegionDesc& r) {
  if ((int)r.sym_start + (int)r.sym_count > sym_dim || (int)r.prb_start + (int)r.prb_count > prb_dim) return;
  const uint8_t ch = r.channel;
  for (int s = 0; s < (int)r.sym_count; ++s) {
    for (int p = 0; p < (int)r.prb_count; ++p) {
      const int gs = (int)r.sym_start + s;
      const int gp = (int)r.prb_start + p;
      chan[(size_t)gs * (size_t)prb_dim + (size_t)gp] = ch;
    }
  }
}

static void scatter_slot_rnti_patch(std::vector<uint32_t>& rnti, int prb_dim, int sym_dim, const SlotRegionDesc& r) {
  if ((int)r.sym_start + (int)r.sym_count > sym_dim || (int)r.prb_start + (int)r.prb_count > prb_dim) return;
  const uint32_t rv = r.rnti;
  for (int s = 0; s < (int)r.sym_count; ++s) {
    for (int p = 0; p < (int)r.prb_count; ++p) {
      const int gs = (int)r.sym_start + s;
      const int gp = (int)r.prb_start + p;
      rnti[(size_t)gs * (size_t)prb_dim + (size_t)gp] = rv;
    }
  }
}

/** Apply ordered regions; payload may be empty (`n_complex` = 0) for **shape-only** records (chan/rnti only, no I/Q bytes). */
static void apply_slot_regions(std::vector<float2>& grid, std::vector<uint8_t>& chan_plane,
                               std::vector<uint32_t>& rnti_plane, int prb_dim, int sym_dim,
                               const std::vector<SlotRegionDesc>& regions, const std::vector<float2>& payload) {
  size_t need = 0;
  for (const auto& r : regions) need += (size_t)r.sym_count * (size_t)r.prb_count;
  const bool shape_only = payload.empty();
  if (!shape_only && need != payload.size()) return;
  if (need == 0u) return;
  size_t off = 0;
  for (const auto& r : regions) {
    const size_t n = (size_t)r.sym_count * (size_t)r.prb_count;
    if (!shape_only)
      scatter_slot_payload_patch(grid, prb_dim, sym_dim, r, payload.data() + off, n);
    scatter_slot_chan_patch(chan_plane, prb_dim, sym_dim, r);
    scatter_slot_rnti_patch(rnti_plane, prb_dim, sym_dim, r);
    if (!shape_only) off += n;
  }
}

static void slot_fifo_finalize(std::vector<float2>& grid, std::vector<uint8_t>& chan_plane,
                               std::vector<uint32_t>& rnti_plane, int ncells, AssembledSlotQueue* outq) {
  GridFrame gf;
  gf.iq = std::move(grid);
  gf.chan = std::move(chan_plane);
  gf.cell_rnti = std::move(rnti_plane);
  grid.assign((size_t)ncells, make_float2(0.f, 0.f));
  chan_plane.assign((size_t)ncells, 0);
  rnti_plane.assign((size_t)ncells, 0u);
  g_stage_output_count_assembly.fetch_add(1, std::memory_order_relaxed);
  outq->push(std::move(gf), "Assembly (slot FIFO)");
}

void slot_message_assembler_main(const std::string& path, int prb_dim, int sym_dim, int ncells,
                                        AssembledSlotQueue* outq) {
  /* One FIFO session: open read end, consume until writer closes (EOF) or error, then close and open again.
     Without this loop, stopping resource_grid_generator.py closes the writer; our reader hits EOF, closes fd,
     and exits — the next writer blocks forever on O_WRONLY with no reader (see generator log:
     "opening slot fifo ... blocks until resgrid_plotter opens read side"). */
  while (g_run.load()) {
    fprintf(stderr, "resgrid_plotter: waiting for slot FIFO writer on %s\n", path.c_str());
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      perror("resgrid_plotter open --slot-fifo");
      g_run = false;
      return;
    }
#if defined(__linux__)
    (void)fcntl(fd, F_SETPIPE_SZ, 1 << 20);
#endif
    uint32_t active_slot = 0xffffffffu;
    std::vector<float2> grid((size_t)ncells, make_float2(0.f, 0.f));
    std::vector<uint8_t> chan_plane((size_t)ncells, 0);
    std::vector<uint32_t> rnti_plane((size_t)ncells, 0u);
    uint64_t bad_slot_order = 0;
    uint64_t slot_id_forward_gap = 0;

    while (g_run.load()) {
      for (;;) {
        if (!g_run.load()) goto slot_fifo_session_done;
        struct pollfd pfd = {fd, POLLIN, 0};
        const int pr = poll(&pfd, 1, 500);
        if (pr < 0) {
          if (errno == EINTR) continue;
          perror("resgrid_plotter poll slot-fifo");
          g_run = false;
          goto slot_fifo_session_done;
        }
        if (pr == 0) continue;
        break;
      }

      SlotMsgHdr h{};
      if (!read_exact_from(fd, &h, sizeof h, "slot-fifo hdr")) goto slot_fifo_session_done;
      if (h.magic != kSlotFifoMagic) continue;
      if (h.version != kSlotProtoVersion1 && h.version != kSlotProtoVersion2) continue;

      std::vector<SlotRegionDesc> regions;
      bool layout_ok = true;

      if (h.version == kSlotProtoVersion2 && (h.flags & kSlotFlagMultiRegion)) {
        uint16_t n_reg = 0;
        if (!read_exact_from(fd, &n_reg, sizeof n_reg, "slot-fifo n_regions")) goto slot_fifo_session_done;
        const size_t rbytes = (size_t)n_reg * sizeof(SlotRegionDesc);
        constexpr size_t kMaxRegionBlob = 1u << 20;
        if (n_reg == 0 || rbytes > kMaxRegionBlob) {
          fprintf(stderr, "resgrid_plotter: slot FIFO invalid n_regions=%u (max %u regions)\n", (unsigned)n_reg,
                  (unsigned)kSlotMaxRegions);
          layout_ok = false;
        } else if (n_reg > kSlotMaxRegions) {
          fprintf(stderr, "resgrid_plotter: slot FIFO n_regions=%u > %u (discard descriptors)\n", (unsigned)n_reg,
                  (unsigned)kSlotMaxRegions);
          std::vector<uint8_t> toss(rbytes);
          if (!read_exact_from(fd, toss.data(), rbytes, "slot-fifo regions (discard)")) goto slot_fifo_session_done;
          layout_ok = false;
        } else {
          regions.resize(n_reg);
          if (!read_exact_from(fd, regions.data(), rbytes, "slot-fifo regions")) goto slot_fifo_session_done;
          uint64_t sum = 0;
          for (const auto& r : regions) {
            sum += (uint64_t)r.sym_count * (uint64_t)r.prb_count;
          }
          if (h.n_complex != 0 && sum != (uint64_t)h.n_complex) {
            fprintf(stderr, "resgrid_plotter: slot FIFO region cell sum %llu != n_complex %u\n",
                    (unsigned long long)sum, (unsigned)h.n_complex);
            layout_ok = false;
          }
          if ((uint64_t)h.n_complex > sum) layout_ok = false;
        }
      } else {
        const uint64_t cells = (uint64_t)h.sym_count * (uint64_t)h.prb_count;
        if (h.n_complex != 0 && cells != (uint64_t)h.n_complex) layout_ok = false;
        if ((uint64_t)h.n_complex > cells) layout_ok = false;
        if (layout_ok) regions.push_back(slot_region_from_hdr(h));
      }

      if ((uint64_t)h.n_complex > (uint64_t)ncells) layout_ok = false;

      std::vector<float2> payload;
      if (h.n_complex > 0) {
        payload.resize((size_t)h.n_complex);
        const size_t nbytes = (size_t)h.n_complex * sizeof(float2);
        if (!read_exact_from(fd, payload.data(), nbytes, "slot-fifo payload")) goto slot_fifo_session_done;
      }

      if (!layout_ok) continue;
      g_assembly_input_msgs.fetch_add(1, std::memory_order_relaxed);

      const uint32_t S = h.slot_id;
      if (active_slot == 0xffffffffu) {
        active_slot = S;
        std::fill(grid.begin(), grid.end(), make_float2(0.f, 0.f));
        std::fill(chan_plane.begin(), chan_plane.end(), 0);
        std::fill(rnti_plane.begin(), rnti_plane.end(), 0u);
        apply_slot_regions(grid, chan_plane, rnti_plane, prb_dim, sym_dim, regions, payload);
        continue;
      }
      if (S == active_slot) {
        apply_slot_regions(grid, chan_plane, rnti_plane, prb_dim, sym_dim, regions, payload);
        continue;
      }
      if (S < active_slot) {
        ++bad_slot_order;
        if (bad_slot_order <= 5u || (bad_slot_order % 1000u) == 0u)
          fprintf(stderr,
                  "resgrid_plotter: WARNING slot FIFO slot_id not consecutive (out-of-order): %u < active %u "
                  "(drop)\n",
                  S, active_slot);
        continue;
      }
      /* S > active_slot: advancing to a new slot. Consecutive stream uses S == active_slot + 1. */
      if (S != active_slot + 1u) {
        ++slot_id_forward_gap;
        if (slot_id_forward_gap <= 5u || (slot_id_forward_gap % 1000u) == 0u)
          fprintf(stderr,
                  "resgrid_plotter: WARNING slot FIFO slot_id not consecutive (gap): new %u after active %u "
                  "(expected %u; missing %u slot id(s))\n",
                  S, active_slot, active_slot + 1u, (unsigned)(S - active_slot - 1u));
      }
      slot_fifo_finalize(grid, chan_plane, rnti_plane, ncells, outq);
      active_slot = S;
      std::fill(grid.begin(), grid.end(), make_float2(0.f, 0.f));
      std::fill(chan_plane.begin(), chan_plane.end(), 0);
      std::fill(rnti_plane.begin(), rnti_plane.end(), 0u);
      apply_slot_regions(grid, chan_plane, rnti_plane, prb_dim, sym_dim, regions, payload);
    }

  slot_fifo_session_done:
    if (active_slot != 0xffffffffu)
      slot_fifo_finalize(grid, chan_plane, rnti_plane, ncells, outq);
    close(fd);
    if (!g_run.load()) return;
    fprintf(stderr,
            "resgrid_plotter: slot FIFO writer disconnected or read ended; will reopen read side when the next "
            "writer opens %s\n",
            path.c_str());
  }
}
static bool read_exact(int fd, void* buf, size_t n) {
  auto* p = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n && g_run.load()) {
    ssize_t r = read(fd, p + got, n - got);
    if (r < 0) {
      perror("read iq-fifo");
      return false;
    }
    if (r == 0) return false;
    got += static_cast<size_t>(r);
  }
  return got == n && g_run.load();
}

/** Open IQ named pipe for read; enlarge buffer on Linux. Returns -1 on failure (perror already). */
int open_iq_fifo_read_end(const char* path, const char* wait_label) {
  fprintf(stderr, "resgrid_plotter: waiting for IQ writer on %s (%s)\n", path, wait_label);
  const int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror("open --iq-fifo");
    return -1;
  }
#if defined(__linux__)
  (void)fcntl(fd, F_SETPIPE_SZ, 1 << 20);
#endif
  return fd;
}

/** Prepare `gf` as zero-chan IQ grid and read one full float32 I/Q raster from `fd`. */
static bool read_one_external_iq_frame(int fd, int ncells, GridFrame& gf) {
  gf.iq.resize((size_t)ncells);
  gf.chan.assign((size_t)ncells, 0);
  gf.cell_rnti.clear();
  return read_exact(fd, gf.iq.data(), (size_t)ncells * sizeof(float2));
}

/** Full-grid FIFO reader: each `read` is one complete I/Q grid (pipeline 2 or channel-mode IQ sidecar). */
void external_grid_frame_reader_main(int fd, int ncells, GridFrameQueue* outq, const char* producer_stage) {
  fprintf(stderr, "resgrid_plotter: external IQ reader → frame queue (cells=%d)\n", ncells);
  while (g_run.load()) {
    GridFrame gf;
    if (!read_one_external_iq_frame(fd, ncells, gf)) break;
    g_assembly_input_msgs.fetch_add(1, std::memory_order_relaxed);
    g_stage_output_count_assembly.fetch_add(1, std::memory_order_relaxed);
    outq->push(std::move(gf), producer_stage);
  }
}

void run_internal_synthetic_producer_loop(InternalFrameQueue* q, void* fps_base, int fps_default, int ncells,
                                                 int prb, int sym) {
  uint64_t fr = 0;
  while (g_run.load()) {
    const int hz = fps_shm_command_hz(fps_base, fps_default);
    const auto period = std::chrono::microseconds(1000000 / hz);
    auto t0 = std::chrono::steady_clock::now();
    GridFrame gf;
    fill_synthetic_iq(gf.iq, gf.chan, gf.cell_rnti, prb, sym, fr++, fps_shm_pattern_u32(fps_base));
    g_assembly_input_msgs.fetch_add(1, std::memory_order_relaxed);
    g_stage_output_count_assembly.fetch_add(1, std::memory_order_relaxed);
    q->push(std::move(gf), "Assembly (internal synthetic)");
    g_producer_frames.fetch_add(1, std::memory_order_relaxed);
    auto t1 = t0 + period;
    std::this_thread::sleep_until(t1);
  }
}

void assembly_stage_main(AssemblyStageCtx* c) {
  fprintf(stderr, "resgrid_plotter: Assembly stage thread started\n");
  if (c->dm == 1) {
    run_internal_synthetic_producer_loop(c->internal_q, c->fps_base, c->fps, c->ncells, c->prb, c->sym);
  } else if (c->dm == 2) {
    external_grid_frame_reader_main(c->iq_fd_full_grid, c->ncells, static_cast<GridFrameQueue*>(c->full_grid_q),
                                    "Assembly (IQ FIFO full-grid)");
  } else {
    std::unique_ptr<std::thread> assembler_th;
    std::unique_ptr<std::thread> sidecar_th;
    if (c->slot_fifo_path && c->slot_assembled_q)
      assembler_th = std::make_unique<std::thread>(slot_message_assembler_main, *c->slot_fifo_path, c->prb, c->sym,
                                                    c->ncells, c->slot_assembled_q);
    if (c->iq_fd_sidecar >= 0 && c->iq_sidecar_q)
      sidecar_th = std::make_unique<std::thread>(external_grid_frame_reader_main, c->iq_fd_sidecar, c->ncells,
                                                 static_cast<GridFrameQueue*>(c->iq_sidecar_q),
                                                 "Assembly (IQ FIFO sidecar)");
    if (assembler_th && assembler_th->joinable()) assembler_th->join();
    if (sidecar_th && sidecar_th->joinable()) sidecar_th->join();
  }
  fprintf(stderr, "resgrid_plotter: Assembly stage thread exiting\n");
}
