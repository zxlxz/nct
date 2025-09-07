#include <cuda_runtime_api.h>
#include <channel_descriptor.h>

#include "nct/cuda/mem.h"

namespace nct::cuda::detail {

auto alloc(usize size, MemType type) -> void* {
  void* ptr = nullptr;

  switch (type) {
    default:
    case MemType::CPU:
      ptr = ::operator new(size);
      if (ptr == nullptr) {
        throw cuda::Error{cudaErrorMemoryAllocation};
      }
      break;
    case MemType::GPU:
      if (auto err = ::cudaMalloc(&ptr, size)) {
        throw cuda::Error{err};
      }
      break;
    case MemType::MIX:
      if (auto err = ::cudaMallocManaged(&ptr, size)) {
        throw cuda::Error{err};
      }
  }
  return ptr;
}

void dealloc(void* ptr, MemType type) {
  if (ptr == nullptr) {
    return;
  }

  switch (type) {
    default:
    case MemType::Heap:    ::operator delete(ptr); break;
    case MemType::Host:    ::cudaFreeHost(ptr); break;
    case MemType::Device:  ::cudaFree(ptr); break;
    case MemType::Managed: ::cudaFree(ptr); break;
  }
}

void memset(void* ptr, u8 val, usize size, stream_t stream) {
  const auto err = stream ? cudaMemsetAsync(ptr, val, size, stream)  //
                          : cudaMemset(ptr, val, size);
  if (err) {
    throw cuda::Error{err};
  }
}

void mcopy(void* dst, const void* src, usize size, stream_t stream) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }

  const auto err = stream ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream)
                          : cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  if (err) {
    throw cuda::Error{err};
  }
}

void msync(void* ptr, usize size, MemType mem_type, stream_t stream) {
  auto loc = cudaMemLocation{
      .type = cudaMemLocationTypeHost,
      .id = 0,
  };

  if (mem_type == MemType::Device) {
    loc.type = cudaMemLocationTypeDevice;
    loc.id = Device::current();
  }

  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, 0, stream)) {
    throw cuda::Error{err};
  }
}

void copy3d(Ptr3D src, Ptr3D dst, stream_t stream) {
  if (src.data == nullptr || dst.data == nullptr) {
    throw cuda::Error{cudaErrorInvalidValue};
  }
  if (src.dims[0] != dst.dims[0] || src.dims[1] != dst.dims[1] || src.dims[2] != dst.dims[2]) {
    throw cuda::Error{cudaErrorInvalidValue};
  }
  if (src.step[0] != 1 || dst.step[1] != 1) {
    throw cuda::Error{cudaErrorInvalidValue};
  }
  if (src.step[1] < src.dims[0] || dst.step[1] < dst.dims[0]) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  auto params = cudaMemcpy3DParms{};
  params.srcPtr = {src.data, src.step[1], src.dims[0], src.dims[1]};
  params.dstPtr = {dst.data, dst.step[1], dst.dims[0], dst.dims[1]};
  params.extent = {src.dims[0], src.dims[1], src.dims[2]};
  params.kind = cudaMemcpyDefault;

  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda::detail
