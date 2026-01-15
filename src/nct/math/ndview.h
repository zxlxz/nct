#pragma once

#include "nct/core.h"

namespace nct::math {

using u32 = unsigned;

template <class T, u32 N = 1>
struct NdView {
  T* _data = nullptr;
  u32 _size[N] = {0};
  u32 _step[N] = {0};
};

template <class T>
struct NdView<T, 1> {
  T* _data = nullptr;
  u32 _size[1] = {0};
  u32 _step[1] = {0};

 public:
  NdView() noexcept = default;

  NdView(T* data, const u32 (&size)[1], const u32 (&step)[1]) noexcept : _data{data}, _size{size[0]}, _step{step[0]} {}

 public:
  auto numel() const noexcept -> u32 {
    return _size[0];
  }

  auto in_bounds(u32 x) const noexcept -> bool {
    return x < _size[0];
  }

  auto operator[](u32 x) const noexcept -> T {
    return _data[x * _step[0]];
  }

  auto operator[](u32 x) noexcept -> T& {
    return _data[x * _step[0]];
  }

  auto is_continuous() const noexcept -> bool {
    return _step[0] == 1;
  }
};

template <class T>
struct NdView<T, 2> {
  static constexpr auto NDIM = 2u;
  T* _data = nullptr;
  u32 _size[NDIM] = {0};
  u32 _step[NDIM] = {0};

 public:
  NdView() noexcept = default;

  NdView(T* data, const u32 (&size)[NDIM], const u32 (&step)[NDIM]) noexcept
      : _data{data}
      , _size{size[0], size[1]}
      , _step{step[0], step[1]} {}

 public:
  auto numel() const noexcept -> u32 {
    return _size[0] * _size[1];
  }

  auto in_bounds(u32 x, u32 y) const noexcept -> bool {
    return x < _size[0] && y < _size[1];
  }

  auto operator[](u32 x, u32 y) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1]];
  }

  auto operator[](u32 x, u32 y) noexcept -> T& {
    return _data[x * _step[0] + y * _step[1]];
  }

  auto slice(const u32 (&idxs)[2], const u32 (&size)[2]) noexcept -> NdView<T, 2> {
    auto offset = idxs[0] * _step[0] + idxs[1] * _step[1];
    return {_data + offset, size, _step};
  }

  auto select(u32 dim, u32 idx) noexcept -> NdView<T, NDIM - 1> {
    auto res = NdView<T, NDIM - 1>{};
    for (auto i = 0u, j = 0u; j < NDIM; ++i) {
      if (i == dim) {
        res._data = _data + idx * _step[i];
        continue;
      } else {
        res._size[j] = _size[i];
        res._step[j] = _step[i];
        ++j;
      }
    }
    return res;
  }

  auto is_continuous() const noexcept -> bool {
    if (_step[0] != 1) {
      return false;
    }
    for (auto i = 1u; i < NDIM - 1; ++i) {
      if (_step[i] != _size[i + 1] * _step[i + 1]) {
        return false;
      }
    }
    return true;
  }
};

template <class T>
struct NdView<T, 3> {
  static constexpr auto NDIM = 3u;
  T* _data = nullptr;
  u32 _size[NDIM] = {0};
  u32 _step[NDIM] = {0};

 public:
  NdView() noexcept = default;

  NdView(T* data, const u32 (&size)[NDIM], const u32 (&step)[NDIM]) noexcept
      : _data{data}
      , _size{size[0], size[1], size[2]}
      , _step{step[0], step[1], step[2]} {}

 public:
  auto numel() const noexcept -> u32 {
    return _size[0] * _size[1] * _size[2];
  }

  auto in_bounds(u32 x, u32 y, u32 z) const noexcept -> bool {
    return x < _size[0] && y < _size[1] && z < _size[2];
  }

  auto operator[](u32 x, u32 y, u32 z) const noexcept -> T {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  auto operator[](u32 x, u32 y, u32 z) noexcept -> T& {
    return _data[x * _step[0] + y * _step[1] + z * _step[2]];
  }

  auto slice(const u32 (&idxs)[3], const u32 (&size)[3]) noexcept -> NdView<T, 3> {
    auto offset = idxs[0] * _step[0] + idxs[1] * _step[1] + idxs[2] * _step[2];
    return {_data + offset, size, _step};
  }

  auto select(u32 dim, u32 idx) noexcept -> NdView<T, NDIM - 1> {
    auto res = NdView<T, NDIM - 1>{};
    for (auto i = 0u, j = 0u; j < NDIM; ++i) {
      if (i == dim) {
        res._data = _data + idx * _step[i];
        continue;
      } else {
        res._size[j] = _size[i];
        res._step[j] = _step[i];
        ++j;
      }
    }
    return res;
  }

  auto is_continuous() const noexcept -> bool {
    if (_step[0] != 1) {
      return false;
    }
    for (auto i = 1u; i < NDIM - 1; ++i) {
      if (_step[i] != _size[i + 1] * _step[i + 1]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace nct::math

namespace nct {
using math::NdView;
}
