/*
 * POSIX mqueue transport: chunked frames + stream-style read_exact for plotter inputs.
 * Built as plain C++ (not NVCC) to avoid libc++/libstdc++ interaction issues under CUDA compilation.
 */
#include "plotter_posix_mq.cuh"

#include <atomic>
#include <mqueue.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern std::atomic<bool> g_run;

/** Strip one chunk header from **payload** (mq message body). Returns false if message too short or bad magic. */
static bool peel_resgrid_chunk(std::vector<uint8_t>& payload, ResgridMqChunkHdr* hdr) {
  if (payload.size() < kResgridMqChunkWireHdrV1Bytes) return false;
  uint32_t magic = 0;
  uint32_t version = 0;
  std::memcpy(&magic, payload.data(), sizeof magic);
  std::memcpy(&version, payload.data() + sizeof magic, sizeof version);
  if (magic != kResgridMqChunkMagic) return false;
  size_t hdr_len = 0;
  if (version == 1u)
    hdr_len = kResgridMqChunkWireHdrV1Bytes;
  else if (version == 2u)
    hdr_len = sizeof(ResgridMqChunkHdr);
  else
    return false;
  if (payload.size() < hdr_len) return false;
  std::memcpy(hdr, payload.data(), hdr_len);
  if (version == 1u) {
    hdr->prb = 0;
    hdr->symbols = 0;
  }
  const size_t body = payload.size() - hdr_len;
  if (body > 0)
    std::memmove(payload.data(), payload.data() + hdr_len, body);
  payload.resize(body);
  return true;
}

static long read_proc_long(const char* path, long fallback) {
  FILE* f = fopen(path, "re");
  if (!f) return fallback;
  long v = fallback;
  if (fscanf(f, "%ld", &v) != 1) v = fallback;
  fclose(f);
  return v;
}

static long effective_msgsize_cap() {
  const char* ev = getenv("RESGRID_MQ_MSGSIZE");
  if (ev && ev[0]) {
    const long v = atol(ev);
    if (v >= 256) return v;
  }
  const long sysmax = read_proc_long("/proc/sys/fs/mqueue/msgsize_max", 8192);
  return sysmax > 0 ? sysmax : 8192;
}

static long effective_maxmsg() {
  const char* ev = getenv("RESGRID_MQ_MAXMSG");
  if (ev && ev[0]) {
    const long v = atol(ev);
    if (v >= 1) return v;
  }
  const long sysmax = read_proc_long("/proc/sys/fs/mqueue/msg_max", 10);
  return sysmax > 0 ? sysmax : 10;
}

bool resgrid_mq_create_queue(const char* mq_name, long maxmsg, long msgsize) {
  if (!mq_name || mq_name[0] != '/') {
    fprintf(stderr, "resgrid_mq_create_queue: invalid mq name %s\n", mq_name ? mq_name : "(null)");
    return false;
  }
  if (maxmsg <= 0) maxmsg = effective_maxmsg();
  if (msgsize <= 0) msgsize = effective_msgsize_cap();
  struct mq_attr attr {};
  attr.mq_flags = 0;
  attr.mq_maxmsg = maxmsg;
  attr.mq_msgsize = msgsize;
  attr.mq_curmsgs = 0;
  (void)mq_unlink(mq_name);
  mqd_t m = mq_open(mq_name, O_CREAT | O_RDONLY, 0666, &attr);
  if (m < (mqd_t)0) {
    fprintf(stderr, "resgrid_mq_create_queue: mq_open(O_CREAT) %s: ", mq_name);
    perror(nullptr);
    return false;
  }
  if (mq_close(m) != 0) {
    perror("resgrid_mq_create_queue: mq_close");
    return false;
  }
  return true;
}

void resgrid_mq_unlink_name(const char* mq_name) {
  if (!mq_name || mq_name[0] != '/') return;
  (void)mq_unlink(mq_name);
}

bool ResgridMqStreamIn::open_read(const char* mq_name) {
  close();
  if (!mq_name || mq_name[0] != '/') {
    fprintf(stderr, "ResgridMqStreamIn::open_read: invalid mq name\n");
    return false;
  }
  mq_ = (int)mq_open(mq_name, O_RDONLY);
  if (mq_ < 0) {
    fprintf(stderr, "resgrid_plotter: mq_open(O_RDONLY) %s: ", mq_name);
    perror(nullptr);
    return false;
  }
  struct mq_attr attr {};
  if (mq_getattr((mqd_t)mq_, &attr) == 0 && attr.mq_msgsize > 0)
    mq_buf_sz_ = (size_t)attr.mq_msgsize;
  else
    mq_buf_sz_ = (size_t)effective_msgsize_cap();
  pending_.clear();
  pending_off_ = 0;
  asm_frame_seq_ = 0xffffffffu;
  asm_logical_len_ = 0;
  asm_buf_.clear();
  asm_highest_chunk_idx_plus1_ = 0;
  asm_wire_version_ = 0;
  asm_saved_prb_ = 0;
  asm_saved_sym_ = 0;
  last_frame_prb_ = 0;
  last_frame_symbols_ = 0;
  return true;
}

void ResgridMqStreamIn::close() {
  if (mq_ >= 0) {
    (void)mq_close((mqd_t)mq_);
    mq_ = -1;
  }
  pending_.clear();
  pending_off_ = 0;
  asm_frame_seq_ = 0xffffffffu;
  asm_logical_len_ = 0;
  asm_buf_.clear();
  asm_highest_chunk_idx_plus1_ = 0;
  asm_wire_version_ = 0;
  asm_saved_prb_ = 0;
  asm_saved_sym_ = 0;
  last_frame_prb_ = 0;
  last_frame_symbols_ = 0;
}

bool ResgridMqStreamIn::mq_receive_one(ResgridMqChunkHdr* hdr, std::vector<uint8_t>& payload) {
  payload.resize(mq_buf_sz_);
  struct timespec ts {};
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_nsec += 500000000;
  if (ts.tv_nsec >= 1000000000) {
    ts.tv_sec += ts.tv_nsec / 1000000000;
    ts.tv_nsec %= 1000000000;
  }
  const ssize_t r =
      mq_timedreceive((mqd_t)mq_, reinterpret_cast<char*>(payload.data()), payload.size(), nullptr, &ts);
  if (r < 0) {
    if (errno == EINTR) return mq_receive_one(hdr, payload);
    if (errno == ETIMEDOUT) {
      std::memset(hdr, 0, sizeof(*hdr));
      payload.clear();
      return true; /* no message */
    }
    perror("resgrid_plotter mq_timedreceive");
    payload.clear();
    return false;
  }
  payload.resize((size_t)r);
  if (!peel_resgrid_chunk(payload, hdr)) {
    std::memset(hdr, 0, sizeof(*hdr));
    payload.clear();
    return true;
  }
  return true;
}

bool ResgridMqStreamIn::pump_one_logical_frame() {
  for (;;) {
    if (!g_run.load()) return false;
    ResgridMqChunkHdr h{};
    std::vector<uint8_t> body;
    if (!mq_receive_one(&h, body)) return false;
    if (h.magic == 0 && body.empty()) continue; /* mq_timedreceive timeout */
    if (h.magic != kResgridMqChunkMagic) continue;
    if (h.version != 1u && h.version != 2u) continue;
    if (h.n_chunks == 0 || h.chunk_idx >= h.n_chunks) continue;
    if (h.logical_len == 0) continue;

    if (h.chunk_idx == 0) {
      asm_frame_seq_ = h.frame_seq;
      asm_logical_len_ = h.logical_len;
      asm_buf_.assign((size_t)h.logical_len, 0);
      asm_highest_chunk_idx_plus1_ = 0;
      asm_wire_version_ = h.version;
      asm_saved_prb_ = h.prb;
      asm_saved_sym_ = h.symbols;
    } else if (h.frame_seq != asm_frame_seq_ || h.logical_len != asm_logical_len_) {
      continue; /* stale / out of sync */
    }
    if (h.chunk_idx != asm_highest_chunk_idx_plus1_) continue;
    if (h.version != asm_wire_version_) continue;
    if (h.version == 2u && (h.prb != asm_saved_prb_ || h.symbols != asm_saved_sym_)) continue;
    if ((uint64_t)h.byte_off + (uint64_t)body.size() > (uint64_t)asm_logical_len_) continue;
    std::memcpy(asm_buf_.data() + h.byte_off, body.data(), body.size());
    asm_highest_chunk_idx_plus1_ = (uint32_t)h.chunk_idx + 1u;
    if (asm_highest_chunk_idx_plus1_ == h.n_chunks) {
      last_frame_prb_ = asm_saved_prb_;
      last_frame_symbols_ = asm_saved_sym_;
      pending_.insert(pending_.end(), asm_buf_.begin(), asm_buf_.end());
      asm_frame_seq_ = 0xffffffffu;
      asm_logical_len_ = 0;
      asm_buf_.clear();
      asm_highest_chunk_idx_plus1_ = 0;
      asm_wire_version_ = 0;
      asm_saved_prb_ = 0;
      asm_saved_sym_ = 0;
      return true;
    }
  }
}

bool ResgridMqStreamIn::read_exact(void* buf, size_t n, const char* ctx) {
  auto* out = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n && g_run.load()) {
    const size_t avail = pending_.size() > pending_off_ ? pending_.size() - pending_off_ : 0;
    if (avail == 0) {
      if (!pump_one_logical_frame()) {
        if (!g_run.load()) return false;
        fprintf(stderr, "resgrid_plotter: MQ read %s: stream ended or error\n", ctx ? ctx : "");
        return false;
      }
      continue;
    }
    const size_t take = std::min(n - got, avail);
    std::memcpy(out + got, pending_.data() + pending_off_, take);
    pending_off_ += take;
    got += take;
    if (pending_off_ == pending_.size()) {
      pending_.clear();
      pending_off_ = 0;
    }
  }
  return got == n && g_run.load();
}

bool resgrid_mq_send_frame(int wq, const void* data, size_t len, uint32_t frame_seq, long mq_msgsize_cap,
                           uint16_t prb, uint16_t symbols) {
  if (wq < 0) return false;
  const size_t hdr_sz = sizeof(ResgridMqChunkHdr);
  if ((size_t)mq_msgsize_cap <= hdr_sz) {
    fprintf(stderr, "resgrid_mq_send_frame: mq_msgsize_cap too small (%ld)\n", mq_msgsize_cap);
    return false;
  }
  const size_t chunk_cap = (size_t)mq_msgsize_cap - hdr_sz;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
  const uint32_t n_chunks_u32 = len == 0 ? 1u : (uint32_t)((len + chunk_cap - 1u) / chunk_cap);
  const uint16_t n_chunks = (uint16_t)std::min<uint32_t>(n_chunks_u32, 65535u);
  if ((uint32_t)n_chunks != n_chunks_u32) {
    fprintf(stderr, "resgrid_mq_send_frame: logical frame too large for chunking\n");
    return false;
  }
  size_t off = 0;
  for (uint16_t ci = 0; ci < n_chunks; ++ci) {
    const size_t remain = len - off;
    const size_t take = std::min(chunk_cap, remain);
    ResgridMqChunkHdr h{};
    h.magic = kResgridMqChunkMagic;
    h.version = kResgridMqChunkVersion;
    h.frame_seq = frame_seq;
    h.chunk_idx = ci;
    h.n_chunks = n_chunks;
    h.logical_len = (uint32_t)len;
    h.byte_off = (uint32_t)off;
    h.prb = prb;
    h.symbols = symbols;
    std::vector<uint8_t> msg;
    msg.resize(hdr_sz + take);
    std::memcpy(msg.data(), &h, hdr_sz);
    if (take) std::memcpy(msg.data() + hdr_sz, p + off, take);
    if (mq_send((mqd_t)wq, reinterpret_cast<const char*>(msg.data()), msg.size(), 0) != 0) {
      fprintf(stderr, "resgrid_mq_send_frame: mq_send chunk %u/%u: ", (unsigned)ci + 1u, (unsigned)n_chunks);
      perror(nullptr);
      return false;
    }
    off += take;
  }
  return true;
}
