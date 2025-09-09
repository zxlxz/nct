#include <cuda_runtime_api.h>
#include <channel_descriptor.h>

#include "nct/cuda/mem.h"

namespace nct {

auto Alloc::alloc(usize size) -> void* {
  auto ptr = static_cast<void*>(nullptr);
  switch (_type) {
    case MemType::CPU:
      if (ptr = ::operator new(size); !ptr) {
        throw cuda::Error{cudaErrorMemoryAllocation};
      }
      break;
    case MemType::GPU:
      if (auto err = ::cudaMalloc(&ptr, size)) {
        throw cuda::Error{err};
      }
      break;
    case MemType::MIXED:
      if (auto err = ::cudaMallocManaged(&ptr, size)) {
        throw cuda::Error{err};
      }
  }

  return ptr;
}

void Alloc::dealloc(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  switch (_type) {
    default:
    case MemType::CPU:   ::operator delete(ptr); break;
    case MemType::GPU:   ::cudaFree(ptr); break;
    case MemType::MIXED: ::cudaFree(ptr); break;
  }
}

void Alloc::sync_cpu(void* ptr, usize size) {
  auto loc = cudaMemLocation{
      .type = cudaMemLocationTypeHost,
      .id = 0,
  };

  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, nullptr)) {
    throw cuda::Error{err};
  }
}

void Alloc::sync_gpu(void* ptr, usize size) {
  auto loc = cudaMemLocation{
      .type = cudaMemLocationTypeDevice,
      .id = cuda::Device::current(),
  };

  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, nullptr)) {
    throw cuda::Error{err};
  }
}

}  // namespace nct

namespace nct::cuda {

void memset(void* ptr, u8 val, usize size, stream_t stream) {
  const auto err = stream ? cudaMemsetAsync(ptr, val, size, stream)  //
                          : cudaMemset(ptr, val, size);
  if (err) {
    throw cuda::Error{err};
  }
}

void memcpy(void* dst, const void* src, usize size, stream_t stream) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }

  const auto err = stream ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream)
                          : cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  if (err) {
    throw cuda::Error{err};
  }
}

template <class T, int N>
static auto make_pitched_ptr(math::NdSlice<T, N> src) -> cudaPitchedPtr {
  static_assert(N == 2 || N == 3, "N must be 2 or 3");

  return cudaPitchedPtr{
      .ptr = src._data,
      .pitch = src._step.y * sizeof(T),
      .xsize = src._dims.x,
      .ysize = src._dims.y,
  };
}

template <class T, int N>
static auto make_copy_extent(math::NdSlice<T, N> src) -> cudaExtent {
  static_assert(N == 2 || N == 3, "N must be 2 or 3");

  auto res = cudaExtent{
      .width = src._dims.x * sizeof(T),
      .height = src._dims.y,
      .depth = 1,
  };
  if constexpr (N == 3) {
    res.depth = src._dims.z;
  }
  return res;
}

template <class T>
void copy2d(math::NdSlice<T, 2> src, math::NdSlice<T, 2> dst, stream_t stream) {
  if (src._data == nullptr || dst._data == nullptr) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._dims != dst._dims) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._step.x != 1 || dst._step.x != 1) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._step.y < src._dims.x || dst._step.y < dst._dims.x) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  const auto params = cudaMemcpy3DParms{
      .srcPtr = make_pitched_ptr(src),
      .dstPtr = make_pitched_ptr(dst),
      .extent = make_copy_extent(src),
      .kind = cudaMemcpyDefault,
  };

  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

template <class T>
void copy3d(math::NdSlice<T, 3> src, math::NdSlice<T, 3> dst, stream_t stream) {
  if (src._data == nullptr || dst._data == nullptr) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._dims != dst._dims) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._step.x != 1 || dst._step.x != 1) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (src._step.y < src._dims.x || dst._step.y < dst._dims.x) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  const auto params = cudaMemcpy3DParms{
      .srcPtr = make_pitched_ptr(src),
      .dstPtr = make_pitched_ptr(dst),
      .extent = make_copy_extent(src),
      .kind = cudaMemcpyDefault,
  };

  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda
