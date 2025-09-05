#pragma once

#include "nct/math/slice.h"
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
auto arr_new(u32 ndim, const u32 dims[], u32 flags) -> arr_t;

void arr_del(arr_t arr);

template <class T>
void arr_set(arr_t arr, const T* src, u32 n, const u32 dims[], const u32 step[], stream_t s);

auto tex_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t;
void tex_del(tex_t obj);
}  // namespace detail

template <class T, int N>
struct Tex;

template <class T, int N = 2>
struct LTex;

template <class T>
struct Tex<T, 2> {
  using dim_t = math::u32x2;
  tex_t _tex;
  dim_t _dim = {0, 0};

 public:
  __device__ auto operator[](math::f32x2 pos) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    tex2D(&res, _tex, pos.x, pos.y);
#endif
    return res;
  }
};

template <class T>
struct Tex<T, 3> {
  using dim_t = math::u32x3;
  tex_t _tex = 0;
  dim_t _dim = {0, 0, 0};

 public:
  __device__ auto operator[](math::f32x3 pos) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    tex3D(&res, _tex, pos.x, pos.y, pos.z);
#endif
    return res;
  }
};

template <class T>
struct LTex<T, 2> {
  using dim_t = math::u32x3;
  tex_t _tex = 0;
  dim_t _dim = {0, 0, 0};

 public:
  struct Layer {
    tex_t _tex = 0;
    u32 _layer = 0;

    __device__ auto operator[](math::f32x2 pos) const -> T {
      auto res = T{0};
#ifdef __CUDACC__
      tex2DLayered(&res, _tex, pos.x, pos.y, _layer);
#endif
      return res;
    }
  };

  __device__ auto operator[](u32 layer) const -> Layer {
    return {_tex, layer};
  }
};

template <class T, int N>
class Texture {
  using dim_t = math::vec<u32, N>;

  arr_t _arr = nullptr;
  tex_t _tex = 0;
  dim_t _dim = {0};

 public:
  Texture() noexcept = default;

  ~Texture() {
    this->reset();
  }

  Texture(const Texture&) = delete;

  Texture& operator=(const Texture&) = delete;

  Texture(Texture&& other) noexcept : _arr{other._arr}, _tex{other._tex}, _dim{other._dim} {
    other._arr = nullptr;
    other._tex = 0;
    other._dim = {0};
  }

  Texture& operator=(Texture&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->reset();
    _arr = other._arr, other._arr = nullptr;
    _tex = other._tex, other._tex = 0;
    _dim = other._dim, other._dim = {0};
    return *this;
  }

  auto operator*() const -> Tex<T, N> {
    return {_tex, _dim};
  }

  static auto with_shape(dim_t dim,
                         FiltMode filt_mode = FiltMode::Point,
                         AddrMode addr_mode = AddrMode::Border) -> Texture {
    auto res = Texture{};
    res._dim = dim;
    res._arr = detail::arr_new<T>(N, &dim.x, 0);
    res._tex = detail::tex_new(res._arr, filt_mode, addr_mode);
    return res;
  }

  void reset() {
    if (_arr == nullptr) {
      return;
    }
    detail::tex_del(_tex);
    detail::arr_del(_arr);
    _tex = 0;
    _arr = nullptr;
  }

  void set_data(math::NdSlice<T, N> data, stream_t stream = nullptr) {
    detail::arr_set(_arr, data._data, &data._dims.x, &data._step.x, stream);
  }
};

template <class T, int N = 2>
class LTexture {
  using dim_t = math::vec<u32, N + 1>;

  arr_t _arr = nullptr;
  tex_t _tex = 0;
  dim_t _dim = {0};

 public:
  LTexture() = default;

  ~LTexture() {
    _tex ? detail::tex_del(_tex) : void();
    _arr ? detail::arr_del(_arr) : void();
  }

  LTexture(const LTexture&) = delete;
  LTexture& operator=(const LTexture&) = delete;

  LTexture(LTexture&& other) noexcept : _arr{other._arr}, _tex{other._tex}, _dim{other._dim} {
    other._arr = nullptr;
    other._tex = 0;
    other._dim = {0};
  }

  LTexture& operator=(LTexture&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->reset();
    _arr = other._arr, other._arr = nullptr;
    _tex = other._tex, other._tex = 0;
    _dim = other._dim, other._dim = {0};
    return *this;
  }

  auto operator*() const -> LTex<T, N> {
    return {_tex, _dim};
  }

  static auto with_dim(dim_t dim,
                       FiltMode filt_mode = FiltMode::Point,
                       AddrMode addr_mode = AddrMode::Border) -> LTexture {
    const auto flags = 1;
    auto res = LTexture{};
    res._dim = dim;
    res._arr = detail::arr_new<T>(N + 1, &dim.x, flags);
    res._tex = detail::tex_new(res._arr, filt_mode, addr_mode);
    return res;
  }

  static auto from_slice(math::NdSlice<T, N + 1> data,
                         FiltMode filt_mode = FiltMode::Point,
                         AddrMode addr_mode = AddrMode::Border) -> LTexture {
    auto res = LTexture::with_dim(data._dims, filt_mode, addr_mode);
    res.set_data(data);
    return res;
  }

  void reset() {
    if (_arr == nullptr) {
      return;
    }
    detail::tex_del(_tex), _tex = 0;
    detail::arr_del(_arr), _arr = nullptr;
  }

  void set_data(math::NdSlice<T, N + 1> src, stream_t stream = nullptr) {
    detail::arr_set(_arr, src._data, N + 1, &src._dims.x, &src._step.x, stream);
  }
};

}  // namespace nct::cuda
