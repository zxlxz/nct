#pragma once

#include "nct/core.h"
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
auto array_new(const size_t (&size)[3], unsigned flags = 0) -> arr_t;
void array_del(arr_t arr);
void array_set(arr_t arr, const void* src);

auto texture_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t;
void texture_del(tex_t obj);

template <class T, unsigned N>
class Array {
  static_assert(N >= 1 && N <= 3, "Array only supports 1D, 2D, and 3D.");

 public:
  arr_t _arr = nullptr;
  tex_t _tex = 0;
  size_t _size[N] = {0};

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

  static auto with_shape(const size_t (&size)[N]) -> Array {
    const size_t fixed_size[3] = {size[0], N > 1 ? size[1] : 1, N > 2 ? size[2] : 1};
    auto res = Array{};
    res._arr = cuda::array_new<T>(fixed_size, 0);
    for (u32 i = 0; i < N; ++i) {
      res._size[i] = static_cast<u32>(size[i]);
    }
    return res;
  }

  static auto from(math::NdView<T, N> src) -> Array {
    auto res = Array::with_shape(src._size);
    res.assign(src);
    return res;
  }

  void assign(math::NdView<T, N> src) {
    const size_t size[3] = {_size[0], N > 1 ? _size[1] : 1, N > 2 ? _size[2] : 1};
    cuda::array_set(_arr, src._data);
  }

  auto tex(FiltMode filt_mode = FiltMode::Point, AddrMode addr_mode = AddrMode::Clamp) -> tex_t {
    if (_tex != 0) {
      cuda::texture_del(_tex);
    }
    _tex = cuda::texture_new(_arr, filt_mode, addr_mode);
    return _tex;
  }
};

template <class T, unsigned N>
class LayeredArray {
  static_assert(N >= 1 && N <= 3, "Array only supports 1D, 2D, and 3D.");

 public:
  arr_t _arr = nullptr;
  tex_t _tex = 0;
  u32 _size[N] = {0};

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

  static auto with_shape(size_t layers, const size_t (&size)[N - 1]) -> LayeredArray {
    const size_t fixed_size[3] = {size[0], N > 2 ? size[1] : 1, layers};
    auto res = LayeredArray{};
    res._arr = cuda::array_new<T>(fixed_size, 1);
    for (u32 i = 0; i < N - 1; ++i) {
      res._size[i] = static_cast<u32>(size[i]);
    }
    res._size[N - 1] = static_cast<u32>(layers);
    return res;
  }

  auto tex(FiltMode filt_mode = FiltMode::Point, AddrMode addr_mode = AddrMode::Clamp) -> tex_t {
    if (_tex != 0) {
      cuda::texture_del(_tex);
    }
    _tex = cuda::texture_new(_arr, static_cast<int>(filt_mode), static_cast<int>(addr_mode));
    return _tex;
  }
};

}  // namespace nct::cuda
