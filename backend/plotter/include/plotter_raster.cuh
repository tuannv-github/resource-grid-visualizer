#pragma once
/** Raster stage: CUDA render contexts, work queue, pool merger. Include only after queue typedefs and `GridFrame` in
 *  `plotter_common.cuh`, and after `plotter_capturer.cuh` (capturer runs parallel to raster → encoder). */

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

class ResgridMqStreamIn;

class CapturerStage;

/** One grid to rasterize in pool mode; `seq` preserves assembler→encoder order at the RGBA queue. */
struct RasterWorkItem {
  uint64_t seq = 0;
  GridFrame gf;
};

/** Bounded queue between single-thread acquirer and raster worker pool (no frame drops). */
class RasterWorkQueue {
 public:
  explicit RasterWorkQueue(int capacity);
  void push(RasterWorkItem&& frame, const char* producer_stage = nullptr);
  bool pop(RasterWorkItem& out, int wait_ms);
  bool try_pop(RasterWorkItem& out);

 private:
  int cap_;
  std::mutex mu_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::deque<RasterWorkItem> q_;
};

/** CUDA raster + optional multi-thread pool; standalone ``--mq-resource-grid`` uses `raster_attach_full_grid_mq_reader` + `raster_run`. */
class RasterStage {
 public:
  void raster_configure(int rt, size_t rgba_bytes, InternalFrameQueue* internal_q, FullGridFifoQueue* full_grid_q,
                        AssembledSlotQueue* assembled_q, GridFrameQueue* sidecar_q, bool resource_grid_slot_stream,
                        RgbaFrameQueue* rgba_out, void* fps_base, int ncells, int prb, int sym, int frame_w, int frame_h,
                        int off_x, int off_y, int content_w, int content_h, float mag_scale, dim3 grid_mag,
                        dim3 block_mag, dim3 grid2, dim3 block2, CapturerStage* capturer);

  void raster_set_thread_cuda(int index, float2* d_z, float* d_mag, uint8_t* d_chan, uint32_t* d_cell_rnti,
                              uint8_t* d_rgba, cudaStream_t cuda_stream);

  /** Pool mode (`rt` > 1): work queue between acquirer and workers; clears per-thread IQ/RGBA queue pointers. */
  void raster_configure_pool(RasterWorkQueue* work_q);

  /** Standalone ``--mq-resource-grid`` (no FAPI): MQ read end owned by raster — spawns MQ→`FullGridFifoQueue` thread alongside consumer(s). Call after `raster_configure`. */
  void raster_attach_full_grid_mq_reader(ResgridMqStreamIn* iq_mq, bool resource_grid_slot_iq_stream);

  /** Single-thread main or pool supervisor (merger + acquirer + workers). */
  void raster_run();

 private:
  struct PerThread {
    InternalFrameQueue* internal_q = nullptr;
    FullGridFifoQueue* full_grid_q = nullptr;
    AssembledSlotQueue* assembled_q = nullptr;
    GridFrameQueue* sidecar_q = nullptr;
    bool resource_grid_slot_stream = false;
    RgbaFrameQueue* rgba_out = nullptr;
    void* fps_base = nullptr;
    int ncells = 0;
    int prb = 0;
    int sym = 0;
    int frame_w = 0;
    int frame_h = 0;
    int off_x = 0;
    int off_y = 0;
    int content_w = 0;
    int content_h = 0;
    float mag_scale = 1.f;
    float2* d_z = nullptr;
    float* d_mag = nullptr;
    uint8_t* d_chan = nullptr;
    uint32_t* d_cell_rnti = nullptr;
    uint8_t* d_rgba = nullptr;
    dim3 grid_mag{};
    dim3 block_mag{};
    dim3 grid2{};
    dim3 block2{};
    cudaStream_t cuda_stream = nullptr;
    CapturerStage* capturer = nullptr;
  };

  void single_thread_main();
  void acquirer_main();
  void worker_main(int thread_index);
  void merger_main();
  void pool_submit_rgba(uint64_t seq, std::vector<uint8_t>&& rgba);

  int rt_ = 1;
  size_t rgba_bytes_ = 0;
  std::vector<PerThread> threads_;

  InternalFrameQueue* acq_internal_q_ = nullptr;
  FullGridFifoQueue* acq_full_grid_q_ = nullptr;
  AssembledSlotQueue* acq_assembled_q_ = nullptr;
  GridFrameQueue* acq_sidecar_q_ = nullptr;
  bool acq_resource_grid_slot_stream_ = false;
  int acq_ncells_ = 0;
  int acq_prb_ = 0;
  int acq_sym_ = 0;

  RasterWorkQueue* work_q_ = nullptr;

  RgbaFrameQueue* merger_rgba_out_ = nullptr;
  CapturerStage* merger_capturer_ = nullptr;
  std::mutex pool_mu_;
  std::condition_variable pool_cv_;
  std::unordered_map<uint64_t, std::vector<uint8_t>> pool_pending_;
  uint64_t pool_next_merged_seq_ = 0;
  uint64_t pool_acquire_issued_end_ = 0;
  bool pool_acquire_done_ = false;

  ResgridMqStreamIn* full_grid_mq_reader_ = nullptr;
  bool full_grid_mq_reader_slot_stream_ = false;
};
