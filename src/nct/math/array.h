#pragma once

#include "nct/math/nview.h"
#include "nct/cuda/mem.h"

namespace nct::math {

using MemType = cuda::MemType;

template <class T, u32 N = 1>
class Array {
  using view_t = NView<T, N>;
  using data_t = cuda::RawBuf<T>;

  data_t _data{};
  view_t _view{};

 public:
  Array() noexcept = default;
  ~Array() noexcept = default;

  Array(Array&& other) noexcept = default;
  Array& operator=(Array&& other) noexcept = default;

  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;

  static auto with_shape(const u32 (&dims)[N], MemType mtype = MemType::CPU) -> Array {
    u32 size = 1U;
    u32 step[N] = {1U};
    for (auto i = 0U; i < N; ++i) {
      size *= dims[i];
      step[i] = i == 0 ? 1 : step[i - 1] * dims[i - 1];
    }

    auto res = Array{};
    res._data = data_t::with_capacity(size, {mtype});
    res._view = view_t::from(res._data.ptr(), dims, step);
    return res;
  }

 public:
  auto operator*() const -> view_t {
    return _view;
  }

  auto data() const -> T* {
    return _view._data;
  }

  auto dims() const -> const auto& {
    return _view._dims;
  }

  auto size() const -> u32 {
    return _view.size();
  }

  void sync_cpu() {
    return _data.sync_cpu();
  }

  void sync_gpu() {
    return _data.sync_gpu();
  }

 public:
  inline auto operator[](u32 idx) const -> T {
    return _view._data[idx];
  }

  inline auto operator[](u32 idx) -> T& {
    return _view._data[idx];
  }

  inline auto operator[](const u32 (&idxs)[N]) const -> T {
    return _view[idxs];
  }

  inline auto operator[](const u32 (&idxs)[N]) -> T& {
    return _view[idxs];
  }

 public:
#ifndef __CUDACC__
  auto as_bytes() const -> sfc::Slice<const u8> {
    const auto ptr = static_cast<const void*>(_data.ptr());
    const auto cap = _data.cap();
    return {static_cast<const u8*>(ptr), cap * sizeof(T)};
  }

  auto as_bytes_mut() -> sfc::Slice<u8> {
    const auto ptr = static_cast<void*>(_data.ptr());
    const auto cap = _data.cap();
    return {static_cast<u8*>(ptr), cap * sizeof(T)};
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    imp.field("dims", _view._dims);
  }
#endif
};

}  // namespace nct::math
