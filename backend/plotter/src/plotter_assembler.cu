/*
 * Assembler stage: slot / FAPI POSIX mq → `AssembledSlotQueue`; IQ mq readers; internal synthetic producer.
 */
#include "plotter_common.cuh"
#include "plotter_assembler.cuh"

#include <cstdio>
#include <cstring>
#include <errno.h>
#include <memory>
#include <thread>
#include <vector>

#include "plotter_posix_mq.cuh"

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
static bool read_exact_from_mq(ResgridMqStreamIn& mq, void* buf, size_t n, const char* ctx) {
  return mq.read_exact(buf, n, ctx);
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

template <class Q>
static void slot_fifo_finalize_to_q(std::vector<float2>& grid, std::vector<uint8_t>& chan_plane,
                                     std::vector<uint32_t>& rnti_plane, int ncells, Q* outq,
                                     const char* producer_tag) {
  GridFrame gf;
  gf.iq = std::move(grid);
  gf.chan = std::move(chan_plane);
  gf.cell_rnti = std::move(rnti_plane);
  grid.assign((size_t)ncells, make_float2(0.f, 0.f));
  chan_plane.assign((size_t)ncells, 0);
  rnti_plane.assign((size_t)ncells, 0u);
  g_stage_output_count_assembler.fetch_add(1, std::memory_order_relaxed);
  outq->push(std::move(gf), producer_tag);
}

static void slot_fifo_finalize(std::vector<float2>& grid, std::vector<uint8_t>& chan_plane,
                               std::vector<uint32_t>& rnti_plane, int ncells, AssembledSlotQueue* outq) {
  slot_fifo_finalize_to_q(grid, chan_plane, rnti_plane, ncells, outq, "Assembler (slot FIFO)");
}

static SlotRegionDesc region_from_resource_grid_slot_hdr(const ResourceGridSlotIqHdr& h) {
  SlotRegionDesc r{};
  r.sym_start = h.sym_start;
  r.sym_count = h.sym_count;
  r.prb_start = h.prb_start;
  r.prb_count = h.prb_count;
  r.rnti = 0;
  r.channel = 0;
  r.reserved[0] = r.reserved[1] = r.reserved[2] = 0;
  return r;
}

static void slot_message_assembler_main(const std::string& path, int prb_dim, int sym_dim, int ncells,
                                        AssembledSlotQueue* outq) {
  /* Re-open read side after errors so a later writer (e.g. restarted generator) can attach again. */
  while (g_run.load()) {
    fprintf(stderr, "resgrid_plotter: waiting for FAPI / slot MQ writer on %s\n", path.c_str());
    ResgridMqStreamIn mq;
    if (!mq.open_read(path.c_str())) {
      perror("resgrid_plotter open --mq-fapi");
      g_run = false;
      return;
    }
    uint32_t active_slot = 0xffffffffu;
    std::vector<float2> grid((size_t)ncells, make_float2(0.f, 0.f));
    std::vector<uint8_t> chan_plane((size_t)ncells, 0);
    std::vector<uint32_t> rnti_plane((size_t)ncells, 0u);
    uint64_t bad_slot_order = 0;
    uint64_t slot_id_forward_gap = 0;

    while (g_run.load()) {
      SlotMsgHdr h{};
      if (!read_exact_from_mq(mq, &h, sizeof h, "slot-mq hdr")) goto slot_fifo_session_done;
      if (h.magic != kSlotFifoMagic) continue;
      if (h.version != kSlotProtoVersion1 && h.version != kSlotProtoVersion2) continue;

      std::vector<SlotRegionDesc> regions;
      bool layout_ok = true;

      if (h.version == kSlotProtoVersion2 && (h.flags & kSlotFlagMultiRegion)) {
        uint16_t n_reg = 0;
        if (!read_exact_from_mq(mq, &n_reg, sizeof n_reg, "slot-fifo n_regions")) goto slot_fifo_session_done;
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
          if (!read_exact_from_mq(mq, toss.data(), rbytes, "slot-fifo regions (discard)")) goto slot_fifo_session_done;
          layout_ok = false;
        } else {
          regions.resize(n_reg);
          if (!read_exact_from_mq(mq, regions.data(), rbytes, "slot-fifo regions")) goto slot_fifo_session_done;
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
        if (!read_exact_from_mq(mq, payload.data(), nbytes, "slot-fifo payload")) goto slot_fifo_session_done;
      }

      if (!layout_ok) continue;
      g_assembler_input_msgs.fetch_add(1, std::memory_order_relaxed);

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
    mq.close();
    if (!g_run.load()) return;
    fprintf(stderr,
            "resgrid_plotter: slot MQ read session ended; will reopen when the next writer opens %s\n",
            path.c_str());
  }
}

void resource_grid_slot_iq_fifo_reader_main(ResgridMqStreamIn* mq_in, int prb_dim, int sym_dim, int ncells,
                                            GridFrameQueue* outq) {
  fprintf(stderr, "resgrid_plotter: resource-grid slot-IQ stream → frame queue (cells=%d)\n", ncells);
  ResgridMqStreamIn& mq = *mq_in;
  uint32_t active_slot = 0xffffffffu;
  std::vector<float2> grid((size_t)ncells, make_float2(0.f, 0.f));
  std::vector<uint8_t> chan_plane((size_t)ncells, 0);
  std::vector<uint32_t> rnti_plane((size_t)ncells, 0u);
  uint64_t bad_slot_order = 0;
  uint64_t slot_id_forward_gap = 0;

  while (g_run.load()) {
    ResourceGridSlotIqHdr h{};
    if (!read_exact_from_mq(mq, &h, sizeof h, "resource-grid-slot-iq hdr")) goto rg_slot_reader_done;
    if (h.magic != kResourceGridSlotIqMagic) continue;

    bool layout_ok = true;
    if ((uint64_t)h.n_complex > (uint64_t)ncells) layout_ok = false;
    if ((int)h.sym_start + (int)h.sym_count > sym_dim || (int)h.prb_start + (int)h.prb_count > prb_dim) layout_ok = false;
    const uint64_t cells = (uint64_t)h.sym_count * (uint64_t)h.prb_count;
    if (h.n_complex > 0 && cells != (uint64_t)h.n_complex) layout_ok = false;
    if ((uint64_t)h.n_complex > cells) layout_ok = false;

    std::vector<float2> payload;
    if (h.n_complex > 0) {
      payload.resize((size_t)h.n_complex);
      const size_t nbytes = (size_t)h.n_complex * sizeof(float2);
      if (!read_exact_from_mq(mq, payload.data(), nbytes, "resource-grid-slot-iq payload")) goto rg_slot_reader_done;
    }

    if (!layout_ok) continue;
    g_assembler_input_msgs.fetch_add(1, std::memory_order_relaxed);

    std::vector<SlotRegionDesc> regions;
    regions.push_back(region_from_resource_grid_slot_hdr(h));

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
                "resgrid_plotter: WARNING resource-grid slot-IQ slot_id out-of-order: %u < active %u (drop)\n", S,
                active_slot);
      continue;
    }
    if (S != active_slot + 1u) {
      ++slot_id_forward_gap;
      if (slot_id_forward_gap <= 5u || (slot_id_forward_gap % 1000u) == 0u)
        fprintf(stderr,
                "resgrid_plotter: WARNING resource-grid slot-IQ slot_id gap: new %u after active %u (expected %u)\n", S,
                active_slot, active_slot + 1u);
    }
    slot_fifo_finalize_to_q(grid, chan_plane, rnti_plane, ncells, outq, "Raster (resource-grid slot IQ)");
    active_slot = S;
    std::fill(grid.begin(), grid.end(), make_float2(0.f, 0.f));
    std::fill(chan_plane.begin(), chan_plane.end(), 0);
    std::fill(rnti_plane.begin(), rnti_plane.end(), 0u);
    apply_slot_regions(grid, chan_plane, rnti_plane, prb_dim, sym_dim, regions, payload);
  }

rg_slot_reader_done:
  if (active_slot != 0xffffffffu)
    slot_fifo_finalize_to_q(grid, chan_plane, rnti_plane, ncells, outq, "Raster (resource-grid slot IQ)");
  fprintf(stderr, "resgrid_plotter: resource-grid slot-IQ reader exiting (raster MQ thread)\n");
}

bool read_one_external_iq_frame(ResgridMqStreamIn* mq, int ncells, GridFrame& gf) {
  gf.iq.resize((size_t)ncells);
  gf.chan.assign((size_t)ncells, 0);
  gf.cell_rnti.clear();
  if (!mq->read_exact(gf.iq.data(), (size_t)ncells * sizeof(float2), "iq-mq full-grid payload")) return false;
  const uint16_t hp = mq->last_frame_prb();
  const uint16_t hs = mq->last_frame_symbols();
  if (hp > 0u && hs > 0u) {
    const uint64_t cells_hdr = (uint64_t)hp * (uint64_t)hs;
    if (cells_hdr != (uint64_t)(unsigned)ncells) {
      static uint64_t nwarn = 0;
      if (++nwarn <= 5u || (nwarn % 1000u) == 0u)
        fprintf(stderr,
                "resgrid_plotter: IQ mq header prb×sym=%ux%u != plotter ncells=%d (check CUDA_IQ_PRB / CUDA_IQ_SYMBOLS)\n",
                (unsigned)hp, (unsigned)hs, ncells);
    }
  }
  return true;
}

/** Full-grid MQ reader: each logical frame is one complete I/Q grid (pipeline 2 or channel-mode IQ sidecar). */
void external_grid_frame_reader_main(ResgridMqStreamIn* mq, int ncells, GridFrameQueue* outq, const char* producer_stage) {
  fprintf(stderr, "resgrid_plotter: external IQ reader → frame queue (cells=%d)\n", ncells);
  while (g_run.load()) {
    GridFrame gf;
    if (!read_one_external_iq_frame(mq, ncells, gf)) break;
    g_assembler_input_msgs.fetch_add(1, std::memory_order_relaxed);
    g_stage_output_count_assembler.fetch_add(1, std::memory_order_relaxed);
    outq->push(std::move(gf), producer_stage);
  }
}

static bool iq_endpoint_is_zmq(const char* path)
{
  if (!path) return false;
  return std::strncmp(path, "ipc://", 6) == 0 || std::strncmp(path, "tcp://", 6) == 0
      || std::strncmp(path, "inproc://", 9) == 0;
}

ResgridMqStreamIn* AssemblerStage::assembler_open_iq_mq_reader(const char* path, const char* wait_label) {
  if (iq_endpoint_is_zmq(path)) {
    fprintf(stderr, "resgrid_plotter: ZeroMQ SUB for IQ %s (%s)\n", path, wait_label);
    auto* z = new ResgridZmqStreamIn();
    if (!z->open_read(path)) {
      fprintf(stderr, "open --mq-resource-grid (zmq) failed\n");
      delete z;
      return nullptr;
    }
    return z;
  }
  fprintf(stderr, "resgrid_plotter: waiting for IQ writer on %s (%s)\n", path, wait_label);
  auto* p = new ResgridMqStreamIn();
  if (!p->open_read(path)) {
    perror("open --mq-resource-grid");
    delete p;
    return nullptr;
  }
  return p;
}

static void run_internal_synthetic_producer_loop(InternalFrameQueue* q, void* fps_base, int fps_default, int ncells,
                                                 int prb, int sym) {
  uint64_t fr = 0;
  while (g_run.load()) {
    const int hz = fps_shm_command_hz(fps_base, fps_default);
    const auto period = std::chrono::microseconds(1000000 / hz);
    auto t0 = std::chrono::steady_clock::now();
    GridFrame gf;
    fill_synthetic_iq(gf.iq, gf.chan, gf.cell_rnti, prb, sym, fr++, fps_shm_pattern_u32(fps_base));
    g_assembler_input_msgs.fetch_add(1, std::memory_order_relaxed);
    g_stage_output_count_assembler.fetch_add(1, std::memory_order_relaxed);
    q->push(std::move(gf), "Assembler (internal synthetic)");
    g_producer_frames.fetch_add(1, std::memory_order_relaxed);
    auto t1 = t0 + period;
    std::this_thread::sleep_until(t1);
  }
}

void AssemblerStage::assembler_configure(InternalFrameQueue* internal_q, void* fps_base, int fps, int ncells, int prb, int sym,
                               AssembledSlotQueue* slot_assembled_q, const std::string* fapi_fifo_path,
                               ChannelModeIqSidecarQueue* iq_sidecar_q, ResgridMqStreamIn* iq_mq_sidecar) {
  internal_q_ = internal_q;
  fps_base_ = fps_base;
  fps_ = fps;
  ncells_ = ncells;
  prb_ = prb;
  sym_ = sym;
  slot_assembled_q_ = slot_assembled_q;
  fapi_fifo_path_ = fapi_fifo_path;
  iq_sidecar_q_ = iq_sidecar_q;
  iq_mq_sidecar_ = iq_mq_sidecar;
}

void AssemblerStage::assembler_run() {
  fprintf(stderr, "resgrid_plotter: Assembler stage thread started\n");
  const bool have_fapi_path = slot_assembled_q_ && fapi_fifo_path_ && !fapi_fifo_path_->empty();
  const bool have_sidecar = have_fapi_path && iq_sidecar_q_ && iq_mq_sidecar_ && iq_mq_sidecar_->ok();
  const bool have_internal = internal_q_ && !have_fapi_path;

  std::vector<std::thread> workers;
  if (have_internal) {
    workers.emplace_back(run_internal_synthetic_producer_loop, internal_q_, fps_base_, fps_, ncells_, prb_, sym_);
  }
  if (have_fapi_path) {
    workers.emplace_back(slot_message_assembler_main, *fapi_fifo_path_, prb_, sym_, ncells_, slot_assembled_q_);
  }
  if (have_sidecar) {
    workers.emplace_back(external_grid_frame_reader_main, iq_mq_sidecar_, ncells_,
                           static_cast<GridFrameQueue*>(iq_sidecar_q_), "Assembler (IQ MQ sidecar)");
  }
  for (auto& t : workers) {
    if (t.joinable()) t.join();
  }
  fprintf(stderr, "resgrid_plotter: Assembler stage thread exiting\n");
}
