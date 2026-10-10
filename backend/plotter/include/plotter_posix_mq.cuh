#pragma once
/** Chunked POSIX message queue transport for RGBA, full-grid I/Q, and slot/FAPI payloads (replaces named FIFOs). */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ResgridMqChunkHdr {
  uint32_t magic;
  uint32_t version;
  uint32_t frame_seq;
  uint16_t chunk_idx;
  uint16_t n_chunks;
  uint32_t logical_len;
  uint32_t byte_off;
  /** v2+: PRB count and symbol count for the logical payload (e.g. full-grid I/Q); 0 if unknown or v1 producer. */
  uint16_t prb;
  uint16_t symbols;
};
static constexpr uint32_t kResgridMqChunkMagic = 0x52474D43u; /* 'RGMC' LE */
static constexpr uint32_t kResgridMqChunkVersion = 2u;
/** v1 wire header is 24 bytes (no prb/symbols); v2 uses full ``ResgridMqChunkHdr``. */
static constexpr size_t kResgridMqChunkWireHdrV1Bytes = 24u;

/** Read-side stream: reassembles chunked mq messages into a byte stream (FIFO-compatible semantics). */
class ResgridMqStreamIn {
 public:
  ResgridMqStreamIn() = default;
  virtual ~ResgridMqStreamIn() { close(); }
  ResgridMqStreamIn(const ResgridMqStreamIn&) = delete;
  ResgridMqStreamIn& operator=(const ResgridMqStreamIn&) = delete;

  bool open_read(const char* mq_name);
  void close();

  /** Blocking read of exactly **n** bytes (uses 500 ms timeouts while waiting), like read(2) on a pipe. */
  virtual bool read_exact(void* buf, size_t n, const char* ctx);

  virtual bool ok() const { return mq_ >= 0; }

  /** Dimensions from the most recently completed logical frame; 0 if unknown. */
  virtual uint16_t last_frame_prb() const { return last_frame_prb_; }
  virtual uint16_t last_frame_symbols() const { return last_frame_symbols_; }
  /** SFN and slot index within that SFN from the latest frame header. False when the header has neither. */
  virtual bool last_frame_timing(uint16_t& sfn, uint16_t& slot) const {
    (void)sfn;
    (void)slot;
    return false;
  }

 private:
  bool pump_one_logical_frame();
  bool mq_receive_one(ResgridMqChunkHdr* hdr, std::vector<uint8_t>& payload);

  int mq_ = -1;
  size_t mq_buf_sz_ = 8192;
  std::vector<uint8_t> pending_{};
  size_t pending_off_ = 0;

  uint32_t asm_frame_seq_ = 0xffffffffu;
  uint32_t asm_logical_len_ = 0;
  std::vector<uint8_t> asm_buf_{};
  uint32_t asm_highest_chunk_idx_plus1_ = 0;
  uint32_t asm_wire_version_ = 0;
  uint16_t asm_saved_prb_ = 0;
  uint16_t asm_saved_sym_ = 0;
  uint16_t last_frame_prb_ = 0;
  uint16_t last_frame_symbols_ = 0;
};

bool resgrid_mq_create_queue(const char* mq_name, long maxmsg, long msgsize);
void resgrid_mq_unlink_name(const char* mq_name);

/** Send one logical frame (may span multiple mq messages). **frame_seq** must change for each frame. */
bool resgrid_mq_send_frame(int wq, const void* data, size_t len, uint32_t frame_seq, long mq_msgsize_cap,
                           uint16_t prb = 0, uint16_t symbols = 0);

/**
 * Resource-grid IQ over ZeroMQ SUB. ``path`` is a full address (``ipc://`` or ``tcp://``).
 * Each message is two parts: a text header (``resgrid_nai seq=1 prb=133 symbols=14 bytes=N sfn=512 slot=7``) and raw float32 IQ.
 * Images and FAPI stay on POSIX mq.
 */
class ResgridZmqStreamIn : public ResgridMqStreamIn {
 public:
  ResgridZmqStreamIn();
  ~ResgridZmqStreamIn() override;

  bool open_read(const char* path);
  bool read_exact(void* buf, size_t n, const char* ctx) override;
  bool ok() const override { return ok_; }
  uint16_t last_frame_prb() const override { return prb_; }
  uint16_t last_frame_symbols() const override { return symbols_; }
  bool last_frame_timing(uint16_t& sfn, uint16_t& slot) const override {
    if (!has_timing_) return false;
    sfn = sfn_;
    slot = slot_;
    return true;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<uint8_t> pending_{};
  size_t pending_off_ = 0;
  uint16_t prb_ = 0;
  uint16_t symbols_ = 0;
  uint16_t sfn_ = 0;
  uint16_t slot_ = 0;
  bool has_timing_ = false;
  bool ok_ = false;

  bool pump_one_frame();
};

/** Connect, read one resource-grid header, then disconnect. Used to size the canvas before CUDA alloc. */
bool resgrid_zmq_peek_grid_dims(const char* endpoint, int timeout_ms, int* prb, int* symbols);
