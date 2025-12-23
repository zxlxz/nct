#include <cuda_runtime_api.h>
#include <channel_descriptor.h>

#include "nct/gpu/mem.h"

namespace nct::gpu {

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

void copy_3d(const void* src, void* dst, const usize dims[3], usize istride, usize ostride, stream_t stream) {
  if (src == nullptr || dst == nullptr) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  const auto params = cudaMemcpy3DParms{
      .srcPtr = {const_cast<void*>(src), istride, dims[0], dims[1]},
      .dstPtr = {dst, ostride, dims[0], dims[1]},
      .extent = {dims[0], dims[1], dims[2]},
      .kind = cudaMemcpyDefault,
  };

  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::gpu
