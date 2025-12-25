#include <cuda_runtime_api.h>
#include <channel_descriptor.h>
#include "nct/cuda/mem.h"

namespace nct::cuda::detail {

auto err_name(int code) -> const char* {
  const auto name = cudaGetErrorName(static_cast<cudaError_t>(code));
  return name;
}

auto alloc_cpu(size_t size) -> void* {
  auto ptr = static_cast<void*>(::malloc(size));
  if (ptr == nullptr) {
    throw cuda::Error{cudaErrorMemoryAllocation};
  }
  return ptr;
}

void dealloc_cpu(void* ptr) {
  if (ptr != nullptr) {
    ::free(ptr);
  }
}

auto alloc_gpu(size_t size) -> void* {
  auto ptr = static_cast<void*>(nullptr);
  const auto err = ::cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    throw cuda::Error{err};
  }
  return ptr;
}

void dealloc_gpu(void* ptr) {
  if (ptr != nullptr) {
    (void)::cudaFree(ptr);
  }
}

auto alloc_uma(size_t size) -> void* {
  auto ptr = static_cast<void*>(nullptr);
  const auto err = ::cudaMallocManaged(&ptr, size);
  if (err != cudaSuccess) {
    throw cuda::Error{err};
  }
  return ptr;
}

void dealloc_uma(void* ptr) {
  if (ptr != nullptr) {
    (void)::cudaFree(ptr);
  }
}

void prefetch_gpu(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto loc = cudaMemLocation{
      .type = cudaMemLocationTypeDevice,
      .id = cuda::Device::current(),
  };

  const auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, nullptr);
  if (err) {
    throw cuda::Error{err};
  }
}

void prefetch_cpu(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto loc = cudaMemLocation{
      .type = cudaMemLocationTypeHost,
      .id = 0,
  };

  const auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, nullptr);
  if (err) {
    throw cuda::Error{err};
  }
}

void write_bytes(void* ptr, u8 val, size_t size) {
  const auto stream = cuda::stream_current();
  const auto err = stream ? cudaMemsetAsync(ptr, val, size, stream) : cudaMemset(ptr, val, size);
  if (err) {
    throw cuda::Error{err};
  }
}

void copy_bytes(const void* src, void* dst, size_t size) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }

  const auto stream = cuda::stream_current();
  const auto err = stream ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream)
                          : cudaMemcpy(dst, src, size, cudaMemcpyDefault);

  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda::detail
