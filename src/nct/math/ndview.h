#pragma once

#ifndef __hd__
#ifdef __host__
#define __hd__ __host__ __device__
#else
#define __hd__
#endif
#endif

namespace nct::math {

using u32 = unsigned;

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

  __hd__ auto get(u32 idx) const noexcept -> T {
    return _data[idx * _step[0]];
  }

  __hd__ void set(u32 idx, T val) noexcept {
    _data[idx * _step[0]] = val;
  }

  __hd__ auto operator[](u32 idx) const noexcept -> T {
    return _data[idx * _step[0]];
  }

  __hd__ auto operator[](u32 idx) noexcept -> T& {
    return _data[idx * _step[0]];
  }

 public:
  auto is_continuous() const noexcept -> bool {
    return _step[0] == 1;
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

  __hd__ auto get(u32 x, u32 y) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1]];
  }

  __hd__ void set(u32 x, u32 y, T val) noexcept {
    _data[x * _step[0] + y * _step[1]] = val;
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

  __hd__ auto get(u32 x, u32 y, u32 z) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  __hd__ void set(u32 x, u32 y, u32 z, T val) noexcept {
    _data[x * _step[0] + y * _step[1] + z * _step[2]] = val;
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
};

}  // namespace nct::math

namespace nct {
using math::NdView;
}
