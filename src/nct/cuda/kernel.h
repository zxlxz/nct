#pragma once

#ifndef __device__
#define __device__
#define __global__
#endif

#ifndef __CUDACC__
struct dim3 {
  unsigned x, y, z;
};
static const auto blockIdx = dim3{1, 1, 1};
static const auto threadIdx = dim3{1, 1, 1};
static const auto blockDim = dim3{1, 1, 1};
static const auto gridDim = dim3{1, 1, 1};
#endif

namespace nct::cuda {

using i16 = short;
using u16 = unsigned short;

using i32 = int;
using u32 = unsigned;

using f32 = float;
using f64 = double;

using tex_t = unsigned long long;

struct dim3 {
  u32 x = 1;
  u32 y = 1;
  u32 z = 1;

 public:
  dim3(u32 x, u32 y = 1, u32 z = 1) : x{x}, y{y}, z{z} {}

  dim3(const u32 (&s)[3]) : x{s[0]}, y{s[1]}, z{s[2]} {}

  operator ::dim3() const {
    return {x, y, z};
  }

  auto operator%(const dim3& ntrd) const -> dim3 {
    const auto blk_size = dim3{
        (x + ntrd.x - 1) / ntrd.x,
        (y + ntrd.y - 1) / ntrd.y,
        (z + ntrd.z - 1) / ntrd.z,
    };
    return blk_size;
  }
};

template <class T, unsigned N>
struct Tex;

template <class T, unsigned N>
struct LTex;

template <class T>
struct Tex<T, 2> {
  tex_t _tex = 0;
  unsigned _size[2] = {};

 public:
  __device__ auto in_bounds(float x, float y) const -> bool {
    return x >= 0 && x < _size[0] && y >= 0 && y < _size[1];
  }

  __device__ auto get(float x, float y) const -> T {
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
  unsigned _size[3] = {};

 public:
  __device__ auto in_bounds(float x, float y, float z) const -> bool {
    return x >= 0 && x < _size[0] && y >= 0 && y < _size[1] && z >= 0 && z < _size[2];
  }

  __device__ auto get(float x, float y, float z) const -> T {
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
  unsigned _size[3] = {};

 public:
  __device__ auto in_bounds(unsigned k, float x, float y) const -> bool {
    return k < _size[2] && x >= 0 && x < _size[0] && y >= 0 && y < _size[1];
  }

  __device__ auto get(unsigned k, float x, float y) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex2DLayered(&res, _tex, x, y, k);
#endif
    return res;
  }
};

void conf_exec(dim3 work_size, dim3 blk_size) {
  (void)work_size;
  (void)blk_size;
}

}  // namespace nct::cuda

#ifndef __CUDACC__
#define CUDA_RUN(f, ...) cuda::conf_exec(__VA_ARGS__), f
#else
#define CUDA_RUN(f, ...) f<<<__VA_ARGS__>>>
#endif
