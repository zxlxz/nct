#pragma once

#include "nct/math/slice.h"

namespace nct::math {

template <class T, u32 N = 1>
class NdArray {
  using dims_t = math::nvec<u32, N>;
  using idxs_t = math::nvec<u32, N>;
  using view_t = math::NdSlice<T, N>;
  using data_t = nct::RawBuf<T>;

  view_t _view{};
  data_t _data{};

 public:
  NdArray() noexcept = default;
  ~NdArray() noexcept = default;

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  NdArray(const NdArray&) = delete;
  NdArray& operator=(const NdArray&) = delete;

  static auto with_dim(dims_t dims, MemType mtype = MemType::CPU) -> NdArray {
    const auto strides = math::make_strides(dims);
    const auto capacity = (&dims.x)[N - 1] * (&strides.x)[N - 1];

    auto res = NdArray{};
    res._data = data_t::with_capacity(capacity, {mtype});
    res._view = view_t{res._data.ptr(), dims, strides};
    return res;
  }

  auto operator*() -> view_t {
    return _view;
  }

  auto operator[](const dims_t& idx) const -> T {
    return _view[idx];
  }

  auto operator[](const dims_t& idx) -> T& {
    return _view[idx];
  }

  auto data() const -> T* {
    return _view._data;
  }

  auto dims() const -> const dims_t& {
    return _view._dims;
  }

  auto size() const -> u32 {
    return _view.size();
  }

  auto slice(dims_t start, dims_t end) const -> view_t {
    return _view.slice(start, end);
  }

  void sync_cpu() {
    return _data.sync_cpu();
  }

  void sync_gpu() {
    return _data.sync_gpu();
  }

#ifndef __CUDACC__
  auto as_bytes_mut() -> sfc::Slice<u8> {
    const auto ptr = static_cast<void*>(_data.ptr());
    const auto cap = _data.cap();
    return {static_cast<u8*>(ptr), cap * sizeof(T)};
  }
#endif

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    imp.field("dims", _view._dims);
  }
};

}  // namespace nct::math
