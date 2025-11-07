#pragma once

#include "nct/math.h"
#include "nct/cuda/mem.h"

struct cudaArray;

namespace nct::cuda {

using arr_t = struct ::cudaArray*;
using tex_t = unsigned long long;

enum class AddrMode {
  Border = 0,
  Clamp = 1,
};

enum class FiltMode {
  Point = 0,
  Linear = 1,
};

namespace detail {
template <class T>
auto arr_new(u32 ndim, const u32 dims[], u32 flags = 0) -> arr_t;

void arr_del(arr_t arr);

template <class T>
void arr_set(arr_t arr, const T* src, u32 n, const u32 dims[], const u32 step[], stream_t s);

auto tex_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t;
void tex_del(tex_t obj);
}  // namespace detail

template <class T, u32 N>
struct Tex {
  using loc_t = math::vec<f32, N>;
  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  auto in_bounds(loc_t p) const -> bool = delete;
  auto operator[](loc_t p) const -> T = delete;
};

template <class T, u32 N>
struct LTex {
  using loc_t = math::vec<f32, N - 1>;
  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  auto in_bounds(u32 layer, loc_t p) const -> bool;
  auto operator()(u32 layer, loc_t p) const -> T;
};

#ifdef __CUDACC__
template <class T>
struct Tex<T, 2> {
  tex_t _tex = 0;
  u32 _dims[2] = {};

 public:
  __device__ auto in_bounds(vec2f p) const -> bool {
    return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1];
  }

  __device__ auto operator[](vec2f p) const -> T {
    auto res = T{0};
    ::tex2D(&res, _tex, p.x, p.y);
    return res;
  }
};

template <class T>
struct Tex<T, 3> {
  tex_t _tex = 0;
  u32 _dims[3] = {};

 public:
  __device__ auto in_bounds(vec3f p) const -> bool {
    return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1] && p.z >= 0 && p.z < _dims[2];
  }

  __device__ auto operator[](vec3f p) const -> T {
    auto res = T{0};
    ::tex3D(&res, _tex, p.x, p.y, p.z);
    return res;
  }
};

template <class T>
struct LTex<T, 3> {
  tex_t _tex = 0;
  u32 _dims[3] = {};

 public:
  __device__ auto in_bounds(u32 layer, vec2f p) const -> bool {
    if (layer >= _dims[2]) {
      return false;
    }
    return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1];
  }

  __device__ auto operator()(u32 layer, vec2f p) const -> T {
    auto res = T{0};
    ::tex2DLayered(&res, _tex, p.x, p.y, layer);
    return res;
  }
};
#endif

template <class T, u32 N>
class TexArr {
  static_assert(N >= 1 && N <= 3, "nct::cuda::TexArr: N out of range[1,3]");

  arr_t _arr = nullptr;
  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  TexArr() noexcept = default;

  ~TexArr() {
    if (_arr != nullptr) {
      detail::tex_del(_tex);
      detail::arr_del(_arr);
    }
  }

  TexArr(const TexArr&) = delete;

  TexArr& operator=(const TexArr&) = delete;

  TexArr(TexArr&& other) noexcept : _arr{other._arr}, _tex{other._tex} {
    other._arr = nullptr;
    other._tex = 0;
  }

  TexArr& operator=(TexArr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    auto tmp = mem::move(*this);
    mem::swap(_arr, other._arr);
    mem::swap(_tex, other._tex);
    mem::swap(_dims, other._dims);
    return *this;
  }

  auto operator*() const -> Tex<T, N> {
    if constexpr (N == 1) {
      return Tex<T, 1>{_tex, {_dims[0]}};
    } else if constexpr (N == 2) {
      return Tex<T, 2>{_tex, {_dims[0], _dims[1]}};
    } else if constexpr (N == 3) {
      return Tex<T, 3>{_tex, {_dims[0], _dims[1], _dims[2]}};
    }
  }

  static auto with_shape(const u32 (&dims)[N],
                         FiltMode filt_mode = FiltMode::Linear,
                         AddrMode addr_mode = AddrMode::Border) -> TexArr {
    auto res = TexArr{};
    res._arr = detail::arr_new<T>(N, dims);
    res._tex = detail::tex_new(res._arr, filt_mode, addr_mode);
    for (auto i = 0U; i < N; ++i) {
      res._dims[i] = dims[i];
    }
    return res;
  }

  static auto from_slice(math::NView<T, N> data,
                         FiltMode filt_mode = FiltMode::Linear,
                         AddrMode addr_mode = AddrMode::Border) -> TexArr {
    auto res = TexArr::with_shape(data._dims, filt_mode, addr_mode);
    res.set_data(data);
    return res;
  }

  void set_data(math::NView<T, N> data, stream_t stream = nullptr) {
    detail::arr_set(_arr, data._data, N, data._dims, data._step, stream);
  }
};

template <class T, u32 N>
class LTexArr {
  static_assert(N >= 2 && N <= 3, "nct::cuda::LTexArr: N out of range[2,3]");
  arr_t _arr = nullptr;
  tex_t _tex = 0;
  u32 _dims[N] = {0};

 public:
  LTexArr() = default;

  ~LTexArr() {
    if (_arr != nullptr) {
      detail::tex_del(_tex);
      detail::arr_del(_arr);
    }
  }

  LTexArr(const LTexArr&) = delete;
  LTexArr& operator=(const LTexArr&) = delete;

  LTexArr(LTexArr&& other) noexcept : _arr{other._arr}, _tex{other._tex} {
    other._arr = nullptr;
    other._tex = 0;
  }

  LTexArr& operator=(LTexArr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    auto tmp = mem::move(*this);
    mem::swap(_arr, other._arr);
    mem::swap(_tex, other._tex);
    mem::swap(_dims, other._dims);
    return *this;
  }

  auto operator*() const -> LTex<T, N> {
    if constexpr (N == 2) {
      return {_tex, {_dims[0], _dims[1]}};
    } else if constexpr (N == 3) {
      return {_tex, {_dims[0], _dims[1], _dims[2]}};
    }
  }

  static auto with_shape(const u32 (&dims)[N],
                         FiltMode filt_mode = FiltMode::Linear,
                         AddrMode addr_mode = AddrMode::Border) -> LTexArr {
    auto res = LTexArr{};
    res._arr = detail::arr_new<T>(N + 1, dims, 1);
    res._tex = detail::tex_new(res._arr, filt_mode, addr_mode);
    return res;
  }

  static auto from_slice(math::NView<T, N> data,
                         FiltMode filt_mode = FiltMode::Linear,
                         AddrMode addr_mode = AddrMode::Border) -> LTexArr {
    auto res = LTexArr::with_shape(data._dims, filt_mode, addr_mode);
    res.set_data(data);
    return res;
  }

  void set_data(math::NView<T, N> src, stream_t stream = nullptr) {
    detail::arr_set(_arr, src._data, N, src._dims, src._step, stream);
  }
};

}  // namespace nct::cuda
