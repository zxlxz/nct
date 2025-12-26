#pragma once

#include "nct/math/ndview.h"
#include "nct/cuda/mem.h"

namespace nct::math {

using cuda::Alloc;

template <class T, usize N = 1, class A = Alloc>
class NdArray {
  NdView<T, N> _view{};
  A _a{};

 public:
  NdArray() noexcept = default;
  ~NdArray() noexcept = default;

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  NdArray(const NdArray&) = delete;
  NdArray& operator=(const NdArray&) = delete;

  static auto with_shape(const usize (&dims)[N], A a = A{}) -> NdArray {
    usize step[N] = {1U};
    for (auto i = 0U; i < N; ++i) {
      step[i] = i == 0 ? 1 : step[i - 1] * dims[i - 1];
    }

    const auto size = step[N - 1] * dims[N - 1];
    const auto data = static_cast<T*>(a.alloc(size * sizeof(T)));

    auto res = NdArray{};
    res._view = NdView<T, N>{data, dims, step};
    res._a = a;
    return res;
  }

 public:
  auto operator*() const -> NdView<T, N> {
    return _view;
  }

  auto data() const -> T* {
    return _view._data;
  }

  auto size() const -> const usize (&)[N] {
    return _view._size;
  }

  auto numel() const -> usize {
    return _view.numel();
  }

  void sync_cpu() {
    const auto n = _view.numel();
    return _a.sync_cpu(_view._data, n * sizeof(T));
  }

  void sync_gpu() {
    const auto n = _view.numel();
    return _a.sync_gpu(_view._data, n * sizeof(T));
  }

 public:
  inline auto operator[](usize idx) const -> T {
    return _view._data[idx];
  }

  inline auto operator[](usize idx) -> T& {
    return _view._data[idx];
  }

  inline auto operator[](const usize (&idxs)[N]) const -> T {
    return _view[idxs];
  }

  inline auto operator[](const usize (&idxs)[N]) -> T& {
    return _view[idxs];
  }

 public:
#ifndef __CUDACC__
  auto as_bytes() const -> Slice<const u8> {
    const auto ptr = static_cast<const void*>(_view._data);
    const auto cnt = _view.numel();
    return {static_cast<const u8*>(ptr), cnt * sizeof(T)};
  }

  auto as_bytes_mut() -> Slice<u8> {
    const auto ptr = static_cast<void*>(_view._data);
    const auto cnt = _view.numel();
    return {static_cast<u8*>(ptr), cnt * sizeof(T)};
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    imp.field("dims", _view._size);
  }
#endif
};

template <class T, usize N>
void fill(NdView<T, N> view, T val) {
  const auto n = view.numel();
  if (val == 0) {
    cuda::write_bytes(view._data, 0, n * sizeof(T));
  } else {
    for (auto i = 0U; i < n; ++i) {
      view._data[i] = val;
    }
  }
}

}  // namespace nct::math

namespace nct {
using math::NdArray;
using math::Alloc;
}  // namespace nct
