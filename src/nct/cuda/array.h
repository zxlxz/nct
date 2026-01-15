#pragma once

#include "nct/math/ndarray.h"

struct cudaArray;

namespace nct::cuda {

using arr_t = struct ::cudaArray*;
using tex_t = unsigned long long;

enum class AddrMode {
  Wrap = 0,    // 0
  Clamp = 1,   // 1
  Mirror = 2,  // 2
  Border = 3   // 3
};

enum class FiltMode {
  Point = 0,
  Linear = 1,
};

template <class T>
auto array_new(u32 ndim, const u32 (&size)[3], u32 flags = 0) -> arr_t;
void array_del(arr_t arr);
void array_set(arr_t arr, const void* src);
void array_ext(arr_t arr, u32 (&size)[3]);

auto texture_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t;
void texture_del(tex_t obj);

template <class T, unsigned N>
struct Tex {
  tex_t _tex = 0;
  unsigned _size[N] = {};
};

template <class T, unsigned N>
struct LTex {
  tex_t _tex = 0;
  unsigned _size[N] = {};
};

template <class T, unsigned N>
class Array {
  static_assert(N >= 1 && N <= 3, "Array only supports 1D, 2D, and 3D.");

 public:
  arr_t _arr = nullptr;
  tex_t _tex = 0;

 public:
  Array() noexcept = default;

  ~Array() noexcept {
    if (_tex != 0) {
      cuda::texture_del(_tex);
    }
    if (_arr != nullptr) {
      cuda::array_del(_arr);
    }
  }

  Array(Array&& other) noexcept : _arr{mem::take(other._arr)}, _tex{mem::take(other._tex)} {}

  Array& operator=(Array&& other) noexcept {
    if (this != &other) {
      mem::swap(_arr, other._arr);
      mem::swap(_tex, other._tex);
    }
    return *this;
  }

  static auto with_shape(const u32 (&size)[N]) -> Array {
    auto res = Array{};
    res._arr = cuda::array_new<T>(N, size, 0);
    return res;
  }

  static auto from(math::NdView<T, N> src) -> Array {
    auto res = Array::with_shape(src._size);
    res.assign(src);
    return res;
  }

  void assign(math::NdView<T, N> src) {
    cuda::array_set(_arr, src._data);
  }

  auto tex(FiltMode filt_mode = FiltMode::Point, AddrMode addr_mode = AddrMode::Clamp) const -> Tex<T, N> {
    if (_tex == 0) {
      const_cast<tex_t&>(_tex) = cuda::texture_new(_arr, filt_mode, addr_mode);
    }

    auto res = cuda::Tex<T, N>{._tex = _tex};
    cuda::array_ext(_arr, res._size);
    return res;
  }
};

template <class T, unsigned N>
class LayeredArray {
  static_assert(N >= 1 && N <= 3, "Array only supports 1D, 2D, and 3D.");

 public:
  arr_t _arr = nullptr;
  tex_t _tex = 0;

 public:
  LayeredArray() noexcept = default;

  ~LayeredArray() noexcept {
    if (_tex != 0) {
      cuda::texture_del(_tex);
    }
    if (_arr != nullptr) {
      cuda::array_del(_arr);
    }
  }

  LayeredArray(LayeredArray&& other) noexcept : _arr{mem::take(other._arr)}, _tex{mem::take(other._tex)} {}

  LayeredArray& operator=(LayeredArray&& other) noexcept {
    if (this != &other) {
      mem::swap(_arr, other._arr);
      mem::swap(_tex, other._tex);
    }
    return *this;
  }

  static auto with_shape(const u32 (&size)[N]) -> LayeredArray {
    auto res = LayeredArray{};
    res._arr = cuda::array_new<T>(N, size, 1);
    return res;
  }

  static auto from(math::NdView<T, N> src) -> LayeredArray {
    auto res = LayeredArray::with_shape(src._size);
    res.assign(src);
    return res;
  }

  void assign(math::NdView<T, N> src) {
    cuda::array_set(_arr, src._data);
  }

  auto tex(FiltMode filt_mode = FiltMode::Point, AddrMode addr_mode = AddrMode::Clamp) const -> LTex<T, N> {
    if (_tex == 0) {
      const_cast<tex_t&>(_tex) = cuda::texture_new(_arr, filt_mode, addr_mode);
    }

    auto res = cuda::LTex<T, N>{._tex = _tex};
    cuda::array_ext(_arr, N, res._size);
    return res;
  }
};

}  // namespace nct::cuda
