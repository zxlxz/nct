#pragma once

#include "nct/cuda/mod.h"
#include "nct/math/slice.h"

struct cudaArray;

namespace nct::cuda {

using arr_t = cudaArray*;

void memset(void* dst, u8 val, usize size, stream_t stream = nullptr);
void memcpy(void* src, const void* dst, usize size, stream_t stream = nullptr);

template <class T>
void copy(const T* src, T* dst, usize size, stream_t stream = nullptr) {
  return memcpy(dst, src, size * sizeof(T), stream);
}

template <class T>
void copy2d(math::NdSlice<T, 2> src, math::NdSlice<T, 2> dst, stream_t stream = nullptr);

template <class T>
void copy3d(math::NdSlice<T, 3> src, math::NdSlice<T, 3> dst, stream_t stream = nullptr);

}  // namespace nct::cuda
