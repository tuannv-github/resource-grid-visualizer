/*
 * Resource-grid IQ reader: ZeroMQ SUB.
 * Connects to the full address passed in (ipc://, tcp://, inproc://).
 * tcp://*:port and tcp://0.0.0.0:port connect as tcp://127.0.0.1:port.
 */
#include "plotter_posix_mq.cuh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <zmq.hpp>

extern std::atomic<bool> g_run;

namespace {

std::string connect_endpoint(const char* bind_addr)
{
  if (!bind_addr) return {};
  const std::string bind(bind_addr);
  if (bind.rfind("tcp://*:", 0) == 0) return "tcp://127.0.0.1:" + bind.substr(8);
  if (bind.rfind("tcp://0.0.0.0:", 0) == 0) return "tcp://127.0.0.1:" + bind.substr(14);
  return bind;
}

struct ParsedResgridHeader {
  uint16_t prb = 0;
  uint16_t symbols = 0;
  uint16_t sfn = 0;
  uint16_t slot = 0;
  bool has_timing = false;
};

bool parse_header(const zmq::message_t& hdr, ParsedResgridHeader& out)
{
  const std::string text(static_cast<const char*>(hdr.data()), hdr.size());
  unsigned seq = 0;
  unsigned p = 0;
  unsigned s = 0;
  unsigned long nbytes = 0;
  char topic[80] = {};
  if (std::sscanf(text.c_str(), "%79s seq=%u prb=%u symbols=%u bytes=%lu", topic, &seq, &p, &s, &nbytes) < 4)
    return false;
  out.prb = static_cast<uint16_t>(p);
  out.symbols = static_cast<uint16_t>(s);
  bool got_sfn = false;
  bool got_slot = false;
  std::istringstream iss(text);
  std::string tok;
  while (iss >> tok) {
    const auto eq = tok.find('=');
    if (eq == std::string::npos || eq + 1 >= tok.size()) continue;
    const std::string key = tok.substr(0, eq);
    const std::string val = tok.substr(eq + 1);
    char* end = nullptr;
    const unsigned long n = std::strtoul(val.c_str(), &end, 10);
    if (end == val.c_str() || n > 65535ul) continue;
    if (key == "sfn") {
      out.sfn = static_cast<uint16_t>(n);
      got_sfn = true;
    } else if (key == "slot") {
      out.slot = static_cast<uint16_t>(n);
      got_slot = true;
    }
  }
  out.has_timing = got_sfn && got_slot;
  return true;
}

} // namespace

bool resgrid_zmq_peek_grid_dims(const char* endpoint, int timeout_ms, int* prb, int* symbols)
{
  if (!endpoint || !endpoint[0] || !prb || !symbols || timeout_ms < 1) return false;
  try {
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::sub);
    sock.set(zmq::sockopt::linger, 0);
    sock.set(zmq::sockopt::rcvtimeo, timeout_ms);
    sock.set(zmq::sockopt::subscribe, "");
    const std::string bind(endpoint);
    std::string ep = bind;
    if (bind.rfind("tcp://*:", 0) == 0) ep = "tcp://127.0.0.1:" + bind.substr(8);
    else if (bind.rfind("tcp://0.0.0.0:", 0) == 0) ep = "tcp://127.0.0.1:" + bind.substr(14);
    sock.connect(ep);
    zmq::message_t header;
    if (!sock.recv(header, zmq::recv_flags::none)) return false;
    if (!sock.get(zmq::sockopt::rcvmore)) return false;
    zmq::message_t body;
    if (!sock.recv(body, zmq::recv_flags::none)) return false;
    ParsedResgridHeader parsed;
    if (!parse_header(header, parsed) || parsed.prb < 1 || parsed.symbols < 1) return false;
    *prb = static_cast<int>(parsed.prb);
    *symbols = static_cast<int>(parsed.symbols);
    return true;
  } catch (const zmq::error_t& e) {
    fprintf(stderr, "resgrid_plotter: ZeroMQ grid-size peek failed: %s\n", e.what());
    return false;
  }
}

struct ResgridZmqStreamIn::Impl {
  zmq::context_t ctx{1};
  zmq::socket_t sub{ctx, zmq::socket_type::sub};
};

ResgridZmqStreamIn::ResgridZmqStreamIn() = default;

ResgridZmqStreamIn::~ResgridZmqStreamIn()
{
  impl_.reset();
}

bool ResgridZmqStreamIn::open_read(const char* path)
{
  ok_ = false;
  impl_.reset();
  pending_.clear();
  pending_off_ = 0;
  prb_ = 0;
  symbols_ = 0;
  sfn_ = 0;
  slot_ = 0;
  has_timing_ = false;
  const std::string endpoint = connect_endpoint(path);
  if (endpoint.rfind("ipc://", 0) != 0 && endpoint.rfind("tcp://", 0) != 0 && endpoint.rfind("inproc://", 0) != 0) {
    std::fprintf(stderr, "resgrid_plotter: IQ ZeroMQ address must be ipc:// or tcp:// (got %s)\n", path ? path : "(null)");
    return false;
  }
  try {
    impl_ = std::make_unique<Impl>();
    impl_->sub.set(zmq::sockopt::linger, 0);
    impl_->sub.set(zmq::sockopt::rcvtimeo, 500);
    impl_->sub.set(zmq::sockopt::subscribe, "");
    impl_->sub.connect(endpoint);
  } catch (const zmq::error_t& e) {
    std::fprintf(stderr, "resgrid_plotter: zmq connect %s failed: %s\n", endpoint.c_str(), e.what());
    impl_.reset();
    return false;
  }
  ok_ = true;
  std::fprintf(stderr, "resgrid_plotter: ZeroMQ SUB connected %s\n", endpoint.c_str());
  return true;
}

bool ResgridZmqStreamIn::pump_one_frame()
{
  if (!impl_) return false;
  for (;;) {
    if (!g_run.load()) return false;
    zmq::message_t first;
    try {
      const auto got = impl_->sub.recv(first, zmq::recv_flags::none);
      if (!got) continue;
      zmq::message_t body;
      const void* bytes = first.data();
      size_t nbytes = first.size();
      if (first.more()) {
        const auto got_body = impl_->sub.recv(body, zmq::recv_flags::none);
        if (!got_body) return false;
        ParsedResgridHeader parsed;
        if (parse_header(first, parsed)) {
          prb_ = parsed.prb;
          symbols_ = parsed.symbols;
          has_timing_ = parsed.has_timing;
          if (parsed.has_timing) {
            sfn_ = parsed.sfn;
            slot_ = parsed.slot;
            static int timing_logs = 0;
            if (timing_logs < 8) {
              ++timing_logs;
              std::fprintf(stderr, "resgrid_plotter: IQ header sfn=%u slot=%u\n",
                           static_cast<unsigned>(sfn_), static_cast<unsigned>(slot_));
            }
          }
        }
        bytes = body.data();
        nbytes = body.size();
      }
      if (nbytes == 0) continue;
      pending_.assign(static_cast<const uint8_t*>(bytes), static_cast<const uint8_t*>(bytes) + nbytes);
      pending_off_ = 0;
      return true;
    } catch (const zmq::error_t& e) {
      if (e.num() == EAGAIN || e.num() == EINTR) continue;
      std::fprintf(stderr, "resgrid_plotter: zmq recv failed: %s\n", e.what());
      return false;
    }
  }
}

bool ResgridZmqStreamIn::read_exact(void* buf, size_t n, const char* ctx)
{
  auto* out = reinterpret_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n && g_run.load()) {
    const size_t avail = pending_.size() > pending_off_ ? pending_.size() - pending_off_ : 0;
    if (avail == 0) {
      if (!pump_one_frame()) {
        if (!g_run.load()) return false;
        std::fprintf(stderr, "resgrid_plotter: ZMQ read %s: stream ended or error\n", ctx ? ctx : "");
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
