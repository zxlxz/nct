#pragma once

#include "nct/core.h"

struct CUstream_st;
struct CUevent_st;

namespace nct::cuda {

using event_t = CUevent_st*;
using stream_t = CUstream_st*;

struct Error {
  int _code = 0;

 public:
  const char* what() const noexcept;
};

struct Device {
  struct Info {
    usize memory_size;
    usize sm_count;
  };

 public:
  static auto count() -> int;
  static auto current() -> int;
  static void set(int);
  static auto info(int id) -> Info;
};

class Event {
  friend class Stream;
  event_t _raw = nullptr;

 public:
  explicit Event();
  ~Event() noexcept;

  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  auto id() const -> event_t;

  void wait();
};

class Stream {
  stream_t _raw = nullptr;

 public:
  explicit Stream();
  ~Stream() noexcept;

  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  auto id() const -> stream_t;

  void sync();

  auto record() -> Event;
  void wait(const Event& event);
};

template <u32 N, class dim3>
static auto make_blk(const u32 (&dim)[N], const dim3& trd) -> dim3 {
  static_assert(N <= 3, "nct::cuda::make_blk: N out of range(max 3)");
  auto res = dim3{1U, 1U, 1U};
  if constexpr (N > 0) {
    res.x = (dim[0] + trd.x - 1) / trd.x;
  } else if constexpr (N > 1) {
    res.y = (dim[1] + trd.y - 1) / trd.y;
  } else if constexpr (N > 2) {
    res.z = (dim[2] + trd.z - 1) / trd.z;
  }
  return res;
}

}  // namespace nct::cuda

#if !defined(__CUDACC__) && !defined(__device__)
#define __device__
#define __global__
#endif

#if defined(__INTELLISENSE__) && !defined(__device_builtin__)
struct dim3 {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;
};
static const auto blockIdx = dim3{1, 1, 1};
static const auto threadIdx = dim3{1, 1, 1};
static const auto blockDim = dim3{1, 1, 1};
static const auto gridDim = dim3{1, 1, 1};
#endif

#ifdef __INTELLISENSE__
#define CUDA_RUN(f, ...) f
#else
#define CUDA_RUN(f, ...) f<<<__VA_ARGS__>>>
#endif
