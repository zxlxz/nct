#include <cuda_runtime_api.h>
#include <channel_descriptor.h>
#include "nct/cuda/array.h"
#include "nct/cuda/stream.h"

namespace nct::cuda {

template <class T>
auto array_new(u32 ndim, const u32 (&size)[3], u32 flags) -> arr_t {
  const auto fmt = cudaCreateChannelDesc<T>();
  const auto ext = cudaExtent{
      ndim > 0 ? size[0] : 1,
      ndim > 1 ? size[1] : 1,
      ndim > 2 ? size[2] : 1,
  };

  auto res = arr_t{nullptr};
  auto err = cudaMalloc3DArray(&res, &fmt, ext, flags);

  if (err) {
    throw cuda::Error{err};
  }
  return res;
}

void array_del(arr_t arr) {
  if (!arr) {
    return;
  }

  if (auto err = ::cudaFreeArray(arr)) {
    throw cuda::Error{err};
  }
}

void array_ext(arr_t arr, u32 ndim, u32 (&size)[3]) {
  if (arr == nullptr) {
    return;
  }

  auto format = cudaChannelFormatDesc{};
  auto extent = cudaExtent{};
  auto flags = 0;
  if (auto err = cudaArrayGetInfo(&format, &extent, &flags, arr)) {
    throw cuda::Error{err};
  }

  if (ndim > 0) {
    size[0] = static_cast<u32>(extent.width);
  }
  if (ndim > 1) {
    size[1] = static_cast<u32>(extent.height);
  }
  if (ndim > 2) {
    size[2] = static_cast<u32>(extent.depth);
  }
}

void array_set(arr_t arr, const void* src) {
  if (arr == nullptr || src == nullptr) {
    return;
  }

  auto format = cudaChannelFormatDesc{};
  auto extent = cudaExtent{};
  auto flags = 0;
  if (auto err = cudaArrayGetInfo(&format, &extent, &flags, arr)) {
    throw cuda::Error{err};
  }

  const auto type_size = (format.x + format.y + format.z + format.w) / 8;

  const auto src_ptr = cudaPitchedPtr{
      .ptr = const_cast<void*>(src),
      .pitch = extent.width * type_size,
      .xsize = extent.width,
      .ysize = extent.height * extent.depth,
  };

  const auto params = cudaMemcpy3DParms{
      .srcPtr = src_ptr,
      .dstArray = arr,
      .extent = extent,
      .kind = cudaMemcpyDefault,
  };

  const auto stream = cuda::stream_current();
  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

auto texture_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t {
  if (!arr) {
    return tex_t{0};
  }

  auto tex_addr = static_cast<cudaTextureAddressMode>(addr_mode);
  auto tex_filt = static_cast<cudaTextureFilterMode>(filt_mode);

  const auto tex_res = cudaResourceDesc{
      .resType = cudaResourceTypeArray,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = cudaTextureDesc{
      .addressMode = {tex_addr, tex_addr, tex_addr},
      .filterMode = tex_filt,
      .readMode = cudaReadModeElementType,
      .normalizedCoords = 0,
  };

  auto tex = cudaTextureObject_t{0};
  if (auto err = cudaCreateTextureObject(&tex, &tex_res, &tex_desc, nullptr)) {
    throw cuda::Error{err};
  }

  return tex;
}

void texture_del(tex_t obj) {
  if (!obj) {
    return;
  }

  if (auto err = cudaDestroyTextureObject(obj)) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda
