#pragma once

#include "nct/core.h"

#ifndef __device__
#define __device__
#endif

namespace nct::cuda {

using tex_t = unsigned long long;

struct dim3 {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;

#ifdef __CUDACC__
  operator ::dim3() const {
    return ::dim3{x, y, z};
  }
#endif
};

template <class T, u32 N>
struct Tex;

template <class T, u32 N>
struct LTex;

template <class T>
struct Tex<T, 2> {
  tex_t _tex = 0;
  u32 _size[2] = {};

 public:
  __hd__ auto in_bounds(float x, float y) const -> bool {
    return x >= 0 && x < _size[0] && y >= 0 && y < _size[1];
  }

  __device__ auto operator()(float x, float y) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex2D(&res, _tex, x, y);
#endif
    return res;
  }
};

template <class T>
struct Tex<T, 3> {
  tex_t _tex = 0;
  u32 _size[3] = {};

 public:
  __hd__ auto in_bounds(float x, float y, float z) const -> bool {
    return x >= 0 && x < _size[0] && y >= 0 && y < _size[1] && z >= 0 && z < _size[2];
  }

  __device__ auto operator()(float x, float y, float z) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex3D(&res, _tex, x, y, z);
#endif
    return res;
  }
};

template <class T>
struct LTex<T, 3> {
  tex_t _tex = 0;
  u32 _size[3] = {};

 public:
  __hd__ auto in_bounds(u32 k, float x, float y) const -> bool {
    return k < _size[2] && x >= 0 && x < _size[0] && y >= 0 && y < _size[1];
  }

  __device__ auto operator()(u32 k, float x, float y) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex2DLayered(&res, _tex, x, y, k);
#endif
    return res;
  }
};

template <uint32_t N>
static auto make_blk(const uint32_t (&dim)[N], const dim3& trd) -> dim3 {
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

static void conf_exec(const auto& blks, const auto& trds) {
  (void)blks;
  (void)trds;
}

}  // namespace nct::cuda

namespace nct {
#ifdef __INTELLISENSE__
static const auto blockIdx = cuda::dim3{1, 1, 1};
static const auto threadIdx = cuda::dim3{1, 1, 1};
static const auto blockDim = cuda::dim3{1, 1, 1};
static const auto gridDim = cuda::dim3{1, 1, 1};
#endif
}  // namespace nct

#ifndef __CUDACC__
#define __device__
#define __global__
#define CUDA_RUN(f, ...) cuda::conf_exec(__VA_ARGS__), f
#else
#define CUDA_RUN(f, ...) f<<<__VA_ARGS__>>>
#endif
