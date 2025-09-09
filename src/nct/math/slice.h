#pragma once

#include "nct/math/vec.h"

namespace nct::math {

inline namespace ranges {
struct ALL {};
}  // namespace ranges

template <class T, int N = 1>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  using dims_t = vec<u32, 1>;
  using idxs_t = vec<u32, 1>;

  T* _data = nullptr;
  dims_t _dims = {};
  dims_t _step = {};

 public:
  auto size() const -> u32 {
    return _dims.x;
  }

  auto dims() const -> dims_t {
    return _dims;
  }

  __hd__ auto in_bounds(idxs_t idxs) const -> bool {
    return (idxs.x < _dims.x);
  }

  __hd__ auto operator[](idxs_t idxs) const -> T {
    return _data[idxs.x * _step.x];
  }

  __hd__ auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x];
  }

  auto slice(idxs_t start, idxs_t end) const -> NdSlice {
    return NdSlice{_data, end - start, _step};
  }
};

template <class T>
struct NdSlice<T, 2> {
  using dims_t = vec<u32, 2>;
  using idxs_t = vec<u32, 2>;

  T* _data = nullptr;
  dims_t _dims = {};
  dims_t _step = {};

 public:
  __hd__ auto size() const -> u32 {
    return _dims.x * _dims.y;
  }

  __hd__ auto dims() const -> dims_t {
    return _dims;
  }

  __hd__ auto in_bounds(idxs_t idxs) const -> bool {
    return idxs < _dims;
  }

  __hd__ auto operator[](idxs_t idxs) const -> T {
    return _data[idxs.x * _step.x + idxs.y * _step.y];
  }

  __hd__ auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y];
  }

  auto slice(idxs_t start, idxs_t end) const -> NdSlice {
    return NdSlice{_data, end - start, _step};
  }

  template <int... I, int M = sizeof...(I)>
  auto select_dims() const -> NdSlice<T, M> {
    const u32 dims[] = {_dims.x, _dims.y};
    const u32 step[] = {_step.x, _step.y};
    return {_data, {dims[I]...}, {step[I]...}};
  }
};

template <class T>
struct NdSlice<T, 3> {
  using dims_t = vec<u32, 3>;
  using idxs_t = vec<u32, 3>;

  T* _data = nullptr;
  dims_t _dims = {};
  dims_t _step = {};

 public:
  auto size() const -> u32 {
    return _dims.x * _dims.y * _dims.z;
  }

  __hd__ auto dims() const -> const dims_t& {
    return _dims;
  }

  __hd__ auto in_bounds(idxs_t idxs) const -> bool {
    return (idxs.x < _dims.x) && (idxs.y < _dims.y) && (idxs.z < _dims.z);
  }

  __hd__ auto operator[](idxs_t idxs) const -> T {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z];
  }

  __hd__ auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z];
  }

  auto slice(idxs_t start, idxs_t end) const -> NdSlice {
    return NdSlice{_data, end - start, _step};
  }

  auto slice(u32 idx, ranges::ALL, ranges::ALL) -> NdSlice<T, 2> {
    return {_data + idx * _step.x, {_dims.y, _dims.z}, {_step.y, _step.z}};
  }

  auto slice(ranges::ALL, u32 idx, ranges::ALL) -> NdSlice<T, 2> {
    return {_data + idx * _step.y, {_dims.x, _dims.z}, {_step.x, _step.z}};
  }

  auto slice(ALL, ALL, u32 idx) -> NdSlice<T, 2> {
    return {_data + idx * _step.z, {_dims.x, _dims.y}, {_step.x, _step.y}};
  }
};

template <int N>
static auto make_step(vec<u32, N> dims) -> vec<u32, N> {
  auto step = vec<u32, N>{1};
  auto* dims_ptr = &dims.x;
  auto* step_ptr = &step.x;
  for (auto i = 1U; i < N; ++i) {
    step_ptr[i] = step_ptr[i - 1] * dims_ptr[i - 1];
  }
  return step;
}

}  // namespace nct::math
