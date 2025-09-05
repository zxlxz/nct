#pragma once

#include "nct/math/vec.h"

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

#ifndef __CUDACC__
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

template <class T, int N, class DIM3>
static auto make_blk(const math::vec<T, N>& dim, const DIM3& trd) -> DIM3 {
  auto res = DIM3{1, 1, 1};
  if constexpr (N >= 1) {
    res.x = (dim.x + trd.x - 1) / trd.x;
  }
  if constexpr (N >= 2) {
    res.y = (dim.y + trd.y - 1) / trd.y;
  }
  if constexpr (N >= 3) {
    res.z = (dim.z + trd.z - 1) / trd.z;
  }
  return res;
}

}  // namespace nct::cuda

#if !defined(__CUDACC__) || defined(__INTELLISENSE__)
#define CUDA_RUN(f, ...) f
#else
#define CUDA_RUN(f, dims, trds, ...) f<<<::nct::cuda::make_blk(dims, trds), trds, __VA_ARGS__>>>
#endif

#if !defined(__CUDACC__) && !defined(__device__)
#define __device__
#define __global__
#define __shared__
#define __constant__
#endif
