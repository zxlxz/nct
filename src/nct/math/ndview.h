#pragma once

#include "nct/core.h"
#include "nct/cuda/mem.h"

namespace nct::math {

template <class T, u32 N = 1>
struct NdView;

template <class T>
struct NdView<T, 1> {
  T* _data = nullptr;
  u32 _size[1] = {0};
  u32 _step[1] = {0};

 public:
  __hd__ NdView() noexcept = default;

  __hd__ NdView(T* data, u32 size) noexcept : _data{data}, _size{size}, _step{1} {}

  __hd__ NdView(T* data, const u32 (&size)[1], const u32 (&step)[1]) noexcept
      : _data(data)
      , _size{size[0]}
      , _step{step[0]} {}

  __hd__ auto numel() const noexcept -> u32 {
    return _size[0];
  }

  __hd__ auto in_bounds(u32 x) const noexcept -> bool {
    return x < _size[0];
  }

  __hd__ auto operator()(u32 idx) const noexcept -> T {
    return _data[idx * _step[0]];
  }

  __hd__ auto operator()(u32 idx) noexcept -> T& {
    return _data[idx * _step[0]];
  }

  __hd__ auto operator[](const u32 (&idx)[1]) const noexcept -> T {
    return _data[idx[0] * _step[0]];
  }

  __hd__ auto operator[](const u32 (&idx)[1]) noexcept -> T& {
    return _data[idx[0] * _step[0]];
  }

 public:
  auto is_continuous() const noexcept -> bool {
    return _step[0] == 1;
  }

  void fill(T val) {
    const auto n = this->numel();
    if (val == 0 && this->is_continuous()) {
      cuda::write_bytes(_data, 0, n * sizeof(T));
    } else {
      for (auto i = 0U; i < n; ++i) {
        _data[i * _step[0]] = val;
      }
    }
  }

  void copy_from(NdView src) {
    const auto n = num::min(this->numel(), src.numel());
    if (this->is_continuous() && src.is_continuous()) {
      cuda::copy_bytes(src._data, this->_data, n * sizeof(T), nullptr);
    } else {
      for (auto i = 0U; i < n; ++i) {
        _data[i * _step[0]] = src._data[i * src._step[0]];
      }
    }
  }
};

template <class T>
struct NdView<T, 2> {
  T* _data = nullptr;
  u32 _size[2] = {0};
  u32 _step[2] = {0};

 public:
  __hd__ NdView() noexcept = default;

  __hd__ NdView(T* data, const u32 (&size)[2], const u32 (&step)[2]) noexcept
      : _data(data)
      , _size{size[0], size[1]}
      , _step{step[0], step[1]} {}

  __hd__ auto numel() const noexcept -> u32 {
    return _size[0] * _size[1];
  }

  __hd__ auto in_bounds(u32 x, u32 y) const noexcept -> bool {
    return (x < _size[0]) && (y < _size[1]);
  }

  __hd__ auto operator()(u32 x, u32 y) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1]];
  }

  __hd__ auto operator()(u32 x, u32 y) noexcept -> T& {
    return _data[x * _step[0] + y * _step[1]];
  }

  __hd__ auto operator[](const u32 (&idxs)[2]) const noexcept -> T {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1]];
  }

  __hd__ auto operator[](const u32 (&idxs)[2]) noexcept -> T& {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1]];
  }

 public:
  __hd__ auto slice(const u32 (&idxs)[2], const u32 (&size)[2]) noexcept -> NdView<T, 2> {
    const auto data = _data + idxs[0] * _step[0] + idxs[1] * _step[1];
    return {data, size, _step};
  }

  template <u32 I>
  __hd__ auto slice_at(u32 idx) noexcept -> NdView<T, 1> {
    if constexpr (I == 0) {
      return {_data + idx * _step[0], {_size[1]}, {_step[1]}};
    } else if constexpr (I == 1) {
      return {_data + idx * _step[1], {_size[0]}, {_step[0]}};
    } else {
      static_assert(I < 2, "I out of range");
    }
  }

 public:
  auto is_continuous() const noexcept -> bool {
    return _step[0] == 1 && _step[1] == _size[0];
  }

  void fill(T val) {
    if (val == 0 && this->is_continuous()) {
      const auto n = this->numel();
      NdView<T, 1>{_data, n}.fill(val);
    } else {
      for (auto y = 0U; y < _size[1]; ++y) {
        this->slice_at<1>(y).fill(val);
      }
    }
  }

  void copy_from(NdView<T, 2> src) {
    const auto n0 = num::min(this->_size[0], src._size[0]);
    const auto n1 = num::min(this->_size[1], src._size[1]);

    if (this->is_continuous() && src.is_continuous()) {
      cuda::copy_bytes(src._data, this->_data, n0 * n1 * sizeof(T));
    } else {
      for (auto i = 0U; i < n1; ++i) {
        auto dst_view = this->slice_at<1>(i);
        auto src_view = src.slice_at<1>(i);
      }
    }
  }
};

template <class T>
struct NdView<T, 3> {
  T* _data = nullptr;
  u32 _size[3] = {0};
  u32 _step[3] = {0};

 public:
  __hd__ NdView() noexcept = default;

  __hd__ NdView(T* data, const u32 (&size)[3], const u32 (&step)[3]) noexcept
      : _data(data)
      , _size{size[0], size[1], size[2]}
      , _step{step[0], step[1], step[2]} {}

  __hd__ auto numel() const noexcept -> u32 {
    return _size[0] * _size[1] * _size[2];
  }

  __hd__ auto in_bounds(u32 x, u32 y, u32 z) const noexcept -> bool {
    return (x < _size[0]) && (y < _size[1]) && (z < _size[2]);
  }

  __hd__ auto operator()(u32 x, u32 y, u32 z) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  __hd__ auto operator()(u32 x, u32 y, u32 z) noexcept -> T& {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  __hd__ auto operator[](const u32 (&idxs)[3]) const noexcept -> T {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2]];
  }

  __hd__ auto operator[](const u32 (&idxs)[3]) noexcept -> T& {
    return _data[idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2]];
  }

 public:
  auto slice(const u32 (&idxs)[3], const u32 (&size)[3]) noexcept -> NdView<T, 3> {
    const auto data = _data + idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2];
    return {data, size, _step};
  }

  template <u32 I>
  auto slice_at(u32 idx) noexcept -> NdView<T, 2> {
    if constexpr (I == 0) {
      return {_data + idx * _step[0], {_size[1], _size[2]}, {_step[1], _step[2]}};
    } else if constexpr (I == 1) {
      return {_data + idx * _step[1], {_size[0], _size[2]}, {_step[0], _step[2]}};
    } else if constexpr (I == 2) {
      return {_data + idx * _step[2], {_size[0], _size[1]}, {_step[0], _step[1]}};
    } else {
      static_assert(I < 3, "DIM out of range");
    }
  }

 public:
  auto is_continuous() const noexcept -> bool {
    return _step[0] == 1 && _step[1] == _size[0] && _step[2] == _size[0] * _size[1];
  }

  void fill(T val) {
    if (val == 0 && this->is_continuous()) {
      const auto n = this->numel();
      NdView<T, 1>{_data, n}.fill(val);
    } else {
      for (auto z = 0U; z < _size[2]; ++z) {
        this->slice_at<2>(z).fill(val);
      }
    }
  }

  void copy_from(NdView<T, 3> src) {
    const auto n0 = num::min(this->_size[0], src._size[0]);
    const auto n1 = num::min(this->_size[1], src._size[1]);
    const auto n2 = num::min(this->_size[2], src._size[2]);

    if (this->is_continuous() && src.is_continuous()) {
      cuda::copy_bytes(src._data, this->_data, n0 * n1 * n2 * sizeof(T), nullptr);
    } else {
      for (auto i = 0U; i < n2; ++i) {
        auto dst_view = this->slice_at<2>(i);
        auto src_view = src.slice_at<2>(i);
        dst_view.copy_from(src_view);
      }
    }
  }
};

}  // namespace nct::math

namespace nct {
using math::NdView;
}
