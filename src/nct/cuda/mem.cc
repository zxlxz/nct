#include <cuda_runtime_api.h>
#include <channel_descriptor.h>

#include "nct/cuda/mem.h"

namespace nct::cuda {

auto alloc(MemType type, usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  auto ptr = static_cast<void*>(nullptr);
  auto err = cudaSuccess;

  switch (type) {
    case MemType::CPU:   ptr = ::malloc(size); break;
    case MemType::GPU:   err = ::cudaMalloc(&ptr, size); break;
    case MemType::MIXED: err = ::cudaMallocManaged(&ptr, size); break;
  }

  if (ptr == nullptr || err != cudaSuccess) {
    throw cuda::Error{err == cudaSuccess ? cudaErrorMemoryAllocation : err};
  }

  return ptr;
}

void dealloc(MemType type, void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  switch (type) {
    default:
    case MemType::CPU:   ::free(ptr); break;
    case MemType::GPU:   ::cudaFree(ptr); break;
    case MemType::MIXED: ::cudaFree(ptr); break;
  }
}

void prefetch(MemType type, void* ptr, usize size) {
  if (type != MemType::CPU && type != MemType::GPU) {
    return;
  }

  const auto is_cpu = type == MemType::CPU;
  const auto cpu_id = 0;
  const auto gpu_id = cuda::Device::current();

  const auto loc = cudaMemLocation{
      .type = is_cpu ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice,
      .id = is_cpu ? cpu_id : gpu_id,
  };

  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, nullptr)) {
    throw cuda::Error{err};
  }
}

void fill_bytes(void* ptr, u8 val, usize size, stream_t stream) {
  const auto err = stream ? cudaMemsetAsync(ptr, val, size, stream) : cudaMemset(ptr, val, size);
  if (err) {
    throw cuda::Error{err};
  }
}

void copy_bytes(void* dst, const void* src, usize size, stream_t stream) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }

  const auto err = stream ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream)
                          : cudaMemcpy(dst, src, size, cudaMemcpyDefault);

  if (err) {
    throw cuda::Error{err};
  }
}

void copy_3d(Ptr3D src, Ptr3D dst, stream_t stream) {
  // check ptr
  if (src.ptr == nullptr || dst.ptr == nullptr || src.ptr == dst.ptr) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  // check type
  if (src.size != dst.size || src.size == 0) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  // check step
  if (src.step[0] != 1 || dst.step[0] != 1) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  // check dims
  for (auto i = 0; i < 3; ++i) {
    if (src.dims[i] == 0 || dst.dims[i] == 0) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
    if (src.dims[i] != dst.dims[i]) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
  }

  const auto src_pitch = src.step[1] * src.size;
  const auto dst_pitch = dst.step[1] * dst.size;

  const auto width_bytes = src.dims[0] * src.size;
  const auto height_elems = src.dims[1];
  const auto depth_elems = src.dims[2];

  const auto params = cudaMemcpy3DParms{
      .srcPtr = {src.ptr, src_pitch, width_bytes, height_elems},
      .dstPtr = {dst.ptr, dst_pitch, width_bytes, height_elems},
      .extent = {width_bytes, height_elems, depth_elems},
      .kind = cudaMemcpyDefault,
  };

  const auto err = stream ? cudaMemcpy3DAsync(¶ms, stream) : cudaMemcpy3D(¶ms);
  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda
