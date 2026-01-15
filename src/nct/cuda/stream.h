#pragma once

#include "nct/core.h"

struct CUstream_st;
struct CUevent_st;

namespace nct::cuda {

using stream_t = CUstream_st*;

auto stream_new() -> stream_t;
void stream_del(stream_t);
void stream_sync(stream_t);


auto stream_current() -> stream_t;
void stream_push(stream_t);
void stream_pop();

class Stream {
  stream_t _raw = nullptr;

 public:
  explicit Stream() : _raw{cuda::stream_new()} {}

  ~Stream() noexcept {
    if (_raw != nullptr) {
      cuda::stream_del(_raw);
    }
  }

  Stream(Stream&& other) noexcept : _raw{mem::take(other._raw)} {}

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      mem::swap(_raw, other._raw);
    }
    return *this;
  }

  auto raw() const -> stream_t {
    return _raw;
  }

  void sync() {
    cuda::stream_sync(_raw);
  }

 public:
  void run(auto& f) {
    cuda::stream_push(_raw);
    f();
    cuda::stream_pop();
  }
};

}  // namespace nct::cuda
