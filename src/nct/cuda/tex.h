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

struct TexMode {
  AddrMode addr_mode = AddrMode::Border;
  FiltMode filt_mode = FiltMode::Linear;
};

template <class T, u32 N>
struct Tex {
  static_assert(N >= 1 && N <= 3, "Tex only supports 1D, 2D, and 3D.");
  using pos_t = math::vec<f32, N>;

  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  __hd__ auto in_bounds(pos_t p) const -> bool {
    if constexpr (N == 1) {
      return p.x >= 0 && p.x < _dims[0];
    } else if constexpr (N == 2) {
      return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1];
    } else if constexpr (N == 3) {
      return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1] && p.z >= 0 && p.z < _dims[2];
    }
  }

  __device__ auto operator[](pos_t p) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    if constexpr (N == 1) {
      ::tex1D(&res, _tex, p.x);
    } else if constexpr (N == 2) {
      ::tex2D(&res, _tex, p.x, p.y);
    } else if constexpr (N == 3) {
      ::tex3D(&res, _tex, p.x, p.y, p.z);
    }
#endif
    return res;
  }
};

template <class T, u32 N>
struct LTex {
  static_assert(N >= 2 && N <= 3, "LTex only supports 2D and 3D.");
  using pos_t = math::vec<f32, N - 1>;

  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  __hd__ auto in_bounds(u32 layer, pos_t p) const -> bool {
    if (layer >= _dims[N - 1]) {
      return false;
    }
    if constexpr (N == 2) {
      return p.x >= 0 && p.x < _dims[0];
    } else if constexpr (N == 3) {
      return p.x >= 0 && p.x < _dims[0] && p.y >= 0 && p.y < _dims[1];
    }
  }

  __device__ auto operator()(u32 layer, pos_t p) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    if constexpr (N == 2) {
      ::tex1DLayered(&res, _tex, p.x, layer);
    } else if constexpr (N == 3) {
      ::tex2DLayered(&res, _tex, p.x, p.y, layer);
    }
#endif
    return res;
  }
};

template <class T, u32 N>
class Texture {
  arr_t _arr = nullptr;
  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  Texture() = default;

  ~Texture() {
    if (_arr != nullptr) {
      detail::tex_del(_tex);
      detail::arr_del(_arr);
    }
  }

  Texture(Texture&& other) noexcept : _arr{mem::take(other._arr)}, _tex{mem::take(other._tex)} {
    for (auto i = 0U; i < N; ++i) {
      _dims[i] = other._dims[i];
    }
  }

  Texture& operator=(Texture&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    auto tmp = static_cast<Texture&&>(*this);
    _arr = mem::take(other._arr);
    _tex = mem::take(other._tex);
    for (auto i = 0U; i < N; ++i) {
      _dims[i] = other._dims[i];
    }
    return *this;
  }

  static auto with_shape(const u32 (&dims)[N], TexMode mode = {}) -> Texture {
    auto res = Texture{};
    res._arr = detail::arr_new<T>(N, dims);
    for (auto i = 0U; i < N; ++i) {
      res._dims[i] = dims[i];
    }
    return res;
  }

  static auto from(math::NView<T, N> data, TexMode mode = {}) -> Texture {
    auto res = Texture::with_shape(data._dims);
    detail::arr_set(res._arr, data._data, N, data._dims, data._step, nullptr);
    return res;
  }

  auto operator*() const -> Tex<T, N> {
    auto res = Tex<T, N>{_tex, {}};
    for (auto i = 0U; i < N; ++i) {
      res._dims[i] = _dims[i];
    }
    return res;
  }
};

template <class T, u32 N>
class LTexture {
  static_assert(N >= 2 && N <= 3, "LTex only supports 2D and 3D.");

  arr_t _arr = nullptr;
  tex_t _tex = 0;
  u32 _dims[N] = {};

 public:
  LTexture() = default;

  ~LTexture() {
    if (_arr != nullptr) {
      detail::tex_del(_tex);
      detail::arr_del(_arr);
    }
  }

  LTexture(LTexture&& other) noexcept : _arr{mem::take(other._arr)}, _tex{mem::take(other._tex)} {
    for (auto i = 0U; i < N; ++i) {
      _dims[i] = other._dims[i];
    }
  }

  LTexture& operator=(LTexture&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    auto tmp = static_cast<LTexture&&>(*this);
    _arr = mem::take(other._arr);
    _tex = mem::take(other._tex);
    for (auto i = 0U; i < N; ++i) {
      _dims[i] = other._dims[i];
    }
    return *this;
  }

  static auto with_shape(const u32 (&dims)[N], TexMode mode = {}) -> LTexture {
    auto res = LTexture{};
    res._arr = detail::arr_new<T>(N, dims, 1);
    res._tex = detail::tex_new(res._arr, mode.filt_mode, mode.addr_mode);
    for (auto i = 0U; i < N; ++i) {
      res._dims[i] = dims[i];
    }
    return res;
  }

  static auto from(math::NView<T, N> data, TexMode mode = {}) -> LTexture {
    auto res = LTexture::with_shape(data._dims);
    detail::arr_set(res._arr, data._data, N, data._dims, data._step, nullptr);
    return res;
  }

  auto operator*() const -> LTex<T, N> {
    auto res = LTex<T, N>{_tex, {}};
    for (auto i = 0U; i < N; ++i) {
      res._dims[i] = _dims[i];
    }
    return res;
  }
};

}  // namespace nct::cuda

namespace nct {
using cuda::Tex;
using cuda::LTex;
}  // namespace nct
