#include <cuda.h>
#include "nct/cuda/mem.h"
#include "nct/cuda/device.h"
#include "nct/cuda/stream.h"

namespace nct::cuda {

auto err_name(int err) -> const char* {
  const auto result = static_cast<CUresult>(err);

  auto err_name = static_cast<const char*>(nullptr);
  if (cuGetErrorName(result, &err_name) != CUDA_SUCCESS) {
    return "UNKNOWN_ERROR";
  }
  return err_name;
}

auto alloc_cpu(size_t size) -> void* {
  auto ptr = static_cast<void*>(::malloc(size));
  if (ptr == nullptr) {
    throw cuda::Error{CUDA_ERROR_OUT_OF_MEMORY};
  }
  return ptr;
}

void dealloc_cpu(void* ptr) {
  if (ptr != nullptr) {
    ::free(ptr);
  }
}

auto alloc_gpu(size_t size) -> void* {
  auto ptr = CUdeviceptr{0};
  if (auto err = ::cuMemAlloc(&ptr, size)) {
    throw cuda::Error{err};
  }
  return __builtin_bit_cast(void*, ptr);
}

void dealloc_gpu(void* ptr) {
  if (ptr != nullptr) {
    auto dptr = __builtin_bit_cast(CUdeviceptr, ptr);
    (void)::cuMemFree(dptr);
  }
}

auto alloc_uma(size_t size) -> void* {
  auto ptr = static_cast<void*>(nullptr);
  if (auto err = ::cuMemAllocManaged(&ptr, size, 0)) {
    throw cuda::Error{err};
  }
  return ptr;
}

void dealloc_uma(void* ptr) {
  const auto dptr = __builtin_bit_cast(CUdeviceptr, ptr);
  if (dptr) {
    (void)::cuMemFree(dptr);
  }
}

void prefetch_gpu(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto dptr = __builtin_bit_cast(CUdeviceptr, ptr);
  const auto loc = CUmemLocation{
      .type = CU_MEM_LOCATION_TYPE_DEVICE,
      .id = Device::current().id,
  };

  const auto err = ::cuMemPrefetchAsync(dptr, size, loc, 0, nullptr);
  if (err) {
    throw cuda::Error{err};
  }
}

void prefetch_cpu(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto loc = CUmemLocation{
      .type = CU_MEM_LOCATION_TYPE_HOST,
      .id = 0,
  };

  const auto err = ::cuMemPrefetchAsync(ptr, size, loc, 0, nullptr);
  if (err) {
    throw cuda::Error{err};
  }
}

void write_bytes(void* ptr, u8 val, size_t size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto dptr = __builtin_bit_cast(CUdeviceptr, ptr);
  const auto stream = cuda::stream_current();
  const auto err = stream ? ::cuMemsetD8Async(dptr, val, size, stream)  // async
                          : ::cuMemsetD8(dptr, val, size);              // sync

  if (err) {
    throw cuda::Error{err};
  }
}

void copy_bytes(const void* src, void* dst, size_t size) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }

  const auto dptr = __builtin_bit_cast(CUdeviceptr, dst);
  const auto sptr = __builtin_bit_cast(CUdeviceptr, src);
  const auto stream = cuda::stream_current();
  const auto err = stream ? ::cuMemcpyAsync(dptr, sptr, size, stream)  // async
                          : ::cuMemcpy(dptr, sptr, size);              // sync

  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda
