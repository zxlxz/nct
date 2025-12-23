#pragma once

#include "nct/math/mod.h"

namespace nct::math {

template <class T, u32 N = 1>
struct NView;

template <class T>
struct NView<T, 1> {
  using idxs_t = u32[1];
  using dims_t = u32[1];

  T* _data = nullptr;
  u32 _dims[1] = {0};
  u32 _step[1] = {0};

 public:
  NView() = default;
  NView(T* data, const dims_t& dims, const idxs_t& step) : _data(data), _dims{dims[0]}, _step{step[0]} {}

  auto size() const -> u32 {
    return _dims[0];
  }

  __hd__ auto in_bounds(const idxs_t& idxs) const -> bool {
    return idxs[0] < _dims[0];
  }

  __hd__ auto operator[](const idxs_t& idxs) const -> T {
    return _data[idxs[0] * _step[0]];
  }

  __hd__ auto operator[](const idxs_t& idxs) -> T& {
    return _data[idxs[0] * _step[0]];
  }

  __hd__ auto operator[](u32 idx) const -> T {
    return _data[idx * _step[0]];
  }

  __hd__ auto operator[](u32 idx) -> T& {
    return _data[idx * _step[0]];
  }
};

template <class T>
struct NView<T, 2> {
  using dims_t = u32[2];
  using idxs_t = u32[2];

  T* _data = nullptr;
  u32 _dims[2] = {0};
  u32 _step[2] = {0};

 public:
  NView() = default;

  NView(T* data, const dims_t& dims, const idxs_t& step)
      : _data(data)
      , _dims{dims[0], dims[1]}
      , _step{step[0], step[1]} {}

  auto size() const -> u32 {
    return _dims[0] * _dims[1];
  }

  __hd__ auto dims() const -> const auto& {
    return _dims;
  }

  __hd__ auto in_bounds(u32 x, u32 y) const -> bool {
    return (x < _dims[0]) && (y < _dims[1]);
  }

  __hd__ auto operator()(u32 x, u32 y) const -> T {
    return _data[x * _step[0] + y * _step[1]];
  }

  __hd__ auto operator()(u32 x, u32 y) -> T& {
    return _data[x * _step[0] + y * _step[1]];
  }

  __hd__ auto operator[](const idxs_t& idxs) const -> T {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1]];
  }

  __hd__ auto operator[](const idxs_t& idxs) -> T& {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1]];
  }

  auto slice(const idxs_t& idxs, const dims_t& dims) -> NView<T, 2> {
    return {_data + idxs[0] * _step[0] + idxs[1] * _step[1], {dims[0], dims[1]}, {_step[0], _step[1]}};
  }

  template <u32 I>
  auto slice_at(u32 idx) const -> NView<T, 1> {
    if constexpr (I == 0) {
      return {_data + idx * _step[0], {_dims[1]}, {_step[1]}};
    } else if constexpr (I == 1) {
      return {_data + idx * _step[1], {_dims[0]}, {_step[0]}};
    } else {
      static_assert(I < 2, "I out of range");
    }
  }
};

template <class T>
struct NView<T, 3> {
  using dims_t = u32[3];
  using idxs_t = u32[3];

  T* _data = nullptr;
  u32 _dims[3] = {0};
  u32 _step[3] = {0};

 public:
  NView() = default;
  NView(T* data, const dims_t& dims, const idxs_t& step)
      : _data(data)
      , _dims{dims[0], dims[1], dims[2]}
      , _step{step[0], step[1], step[2]} {}

  __hd__ auto size() const -> u32 {
    return _dims[0] * _dims[1] * _dims[2];
  }

  __hd__ auto dims() const -> const auto& {
    return _dims;
  }

  __hd__ auto in_bounds(u32 x, u32 y, u32 z) const -> bool {
    return (x < _dims[0]) && (y < _dims[1]) && (z < _dims[2]);
  }

  __hd__ auto operator()(u32 x, u32 y, u32 z) const -> T {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  __hd__ auto operator()(u32 x, u32 y, u32 z) -> T& {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  __hd__ auto operator[](const idxs_t& idxs) const -> T {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2]];
  }

  __hd__ auto operator[](const idxs_t& idxs) -> T& {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2]];
  }

  template <u32 I>
  auto slice_at(u32 idx) -> NView<T, 2> {
    if constexpr (I == 0) {
      return {_data + idx * _step[0], {_dims[1], _dims[2]}, {_step[1], _step[2]}};
    } else if constexpr (I == 1) {
      return {_data + idx * _step[1], {_dims[0], _dims[2]}, {_step[0], _step[2]}};
    } else if constexpr (I == 2) {
      return {_data + idx * _step[2], {_dims[0], _dims[1]}, {_step[0], _step[1]}};
    } else {
      static_assert(I < 3, "DIM out of range");
    }
  }
};

}  // namespace nct::math

namespace nct {
using math::NView;
}
