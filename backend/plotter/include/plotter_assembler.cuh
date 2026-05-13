#pragma once
/** Assembler stage: IQ / FAPI FIFO readers and slot assembler feeding grid queues. Include only after queue typedefs in
 *  `plotter_common.cuh`. */

#include <string>

/** FIFO readers + slot assembler → grid queues (one `run()` thread from pipeline). */
class AssemblerStage {
 public:
  void assembler_configure(InternalFrameQueue* internal_q, void* fps_base, int fps, int ncells, int prb, int sym,
                           AssembledSlotQueue* slot_assembled_q, const std::string* fapi_fifo_path,
                           ChannelModeIqSidecarQueue* iq_sidecar_q, int iq_fd_sidecar);

  /** Starts internal producer / FIFO reader threads and joins them (blocks until all exit). */
  void assembler_run();

  /** Open IQ named pipe read end; used by pipeline before `assembler_run()`. */
  static int assembler_open_iq_fifo_read_end(const char* path, const char* wait_label);

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
  int iq_fd_sidecar_ = -1;
};
