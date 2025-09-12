#pragma once

#include "nct/math/slice.h"

namespace nct::math {

template <class T, int N = 1>
class NdArray {
  using Buf = nct::RawBuf<T>;
  using Inn = math::NdSlice<T, N>;
  using Dim = math::vec<u32, N>;

  Inn _inn{};
  Buf _buf{};

 public:
  NdArray() noexcept = default;
  ~NdArray() noexcept = default;

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  NdArray(const NdArray&) = delete;
  NdArray& operator=(const NdArray&) = delete;

  static auto with_dim(Dim dims, MemType mtype = MemType::CPU) -> NdArray {
    const auto step = math::make_step(dims);
    const auto capacity = (&dims.x)[N - 1] * (&step.x)[N - 1];

    auto res = NdArray{};
    res._buf = Buf::with_capacity(capacity, {mtype});
    res._inn = Inn{res._buf.ptr(), dims, step};
    return res;
  }

  auto operator*() const -> Inn {
    return _inn;
  }

  auto operator[](const Dim& idx) const -> T {
    return _inn[idx];
  }

  auto operator[](const Dim& idx) -> T& {
    return _inn[idx];
  }

  auto data() const -> T* {
    return _inn._data;
  }

  auto dims() const -> const Dim& {
    return _inn._dims;
  }

  auto step() const -> const Dim& {
    return _inn._step;
  }

  auto size() const -> u32 {
    return _inn.size();
  }

  auto slice(Dim start, Dim end) const -> Inn {
    return _inn.slice(start, end);
  }

  void sync_cpu() {
    return _buf.sync_cpu();
  }

  void sync_gpu() {
    return _buf.sync_gpu();
  }
};

}  // namespace nct::math
