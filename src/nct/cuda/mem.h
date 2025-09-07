#pragma once

#include "nct/cuda/mod.h"
#include "nct/math/slice.h"

struct cudaArray;

namespace nct::cuda {

using arr_t = cudaArray*;

enum class MemType {
  CPU = 0,
  GPU = 1,
  MIXED = 2,
};

namespace detail {
auto alloc(usize size, MemType type) -> void*;
void dealloc(void* ptr, MemType type);
void memset(void* ptr, u8 val, usize size, stream_t stream = nullptr);
void msync(void* ptr, usize size, MemType type, stream_t stream = nullptr);
void mcopy(void* src, const void* dst, usize size, stream_t stream = nullptr);

struct Ptr3D {
  void* data = nullptr;
  usize dims[3] = {0, 0, 0};
  usize step[3] = {0, 0, 0};
};
void copy3d(Ptr3D src, Ptr3D dst, stream_t stream = nullptr);
}  // namespace detail

template <class T>
void copy(const T* src, T* dst, usize size, stream_t stream = nullptr) {
  return cuda::detail::mcopy(src, dst, size * sizeof(T), stream);
}

template <class T>
void copy2d(math::NdSlice<T, 2> src, math::NdSlice<T, 2> dst, stream_t stream = nullptr) {
  const auto ps = detail::Ptr3D{
      src._data,
      {sizeof(T) * src._dims.x, src._dims.y, 1U},
      {sizeof(T) * src._step.x, src._step.y, 1U},
  };
  const auto pd = detail::Ptr3D{
      dst._data,
      {sizeof(T) * dst._dims.x, dst._dims.y, 1U},
      {sizeof(T) * dst._step.x, dst._step.y, 1U},
  };
  return cuda::detail::copy3d(ps, pd, stream);
}

template <class T>
void copy3d(math::NdSlice<T, 3> src, math::NdSlice<T, 3> dst, stream_t stream = nullptr) {
  const auto ps = detail::Ptr3D{
      src._data,
      {sizeof(T) * src._dims.x, src._dims.y, src._dims.z},
      {sizeof(T) * src._step.x, src._step.y, src._step.z},
  };
  const auto pd = detail::Ptr3D{
      dst._data,
      {sizeof(T) * dst._dims.x, dst._dims.y, dst._dims.z},
      {sizeof(T) * dst._step.x, dst._step.y, dst._step.z},
  };
  return cuda::detail::copy3d(ps, pd, stream);
}

}  // namespace nct::cuda
