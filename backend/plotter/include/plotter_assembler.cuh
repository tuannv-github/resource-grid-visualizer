#pragma once
/** Assembler stage: IQ / FAPI POSIX mq readers and slot assembler feeding grid queues. Include only after queue typedefs
 *  in `plotter_common.cuh`. */

#include <string>

class ResgridMqStreamIn;

/** MQ readers + slot assembler → grid queues (one `run()` thread from pipeline). */
class AssemblerStage {
 public:
  void assembler_configure(InternalFrameQueue* internal_q, void* fps_base, int fps, int ncells, int prb, int sym,
                           AssembledSlotQueue* slot_assembled_q, const std::string* fapi_fifo_path,
                           ChannelModeIqSidecarQueue* iq_sidecar_q, ResgridMqStreamIn* iq_mq_sidecar);

  /** Starts internal producer / MQ reader threads and joins them (blocks until all exit). */
  void assembler_run();

  /** Open IQ POSIX mq read end; owned by caller (typically `std::unique_ptr` in `plotter_pipeline.cu`). */
  static ResgridMqStreamIn* assembler_open_iq_mq_reader(const char* path, const char* wait_label);

 private:
  InternalFrameQueue* internal_q_ = nullptr;
  void* fps_base_ = nullptr;
  int fps_ = 200;
  int ncells_ = 0;
  int prb_ = 0;
  int sym_ = 0;
  AssembledSlotQueue* slot_assembled_q_ = nullptr;
  const std::string* fapi_fifo_path_ = nullptr;
  ChannelModeIqSidecarQueue* iq_sidecar_q_ = nullptr;
  ResgridMqStreamIn* iq_mq_sidecar_ = nullptr;
};
