
#include <cuda_runtime_api.h>
#include "nct/cuda/tex.h"

namespace nct::cuda::detail {

template <class T>
auto arr_new(u32 ndim, const u32 dims[], u32 flags) -> arr_t {
  if (ndim <= 0 || ndim >= 3) {
    throw cuda::Error{cudaErrorInvalidValue};
  }

  const auto fmt = cudaCreateChannelDesc<T>();
  const auto ext = cudaExtent{
      ndim > 0 ? dims[0] : 1,
      ndim > 1 ? dims[1] : 0,
      ndim > 2 ? dims[2] : 0,
  };

  auto res = arr_t{nullptr};
  auto err = cudaMalloc3DArray(&res, &fmt, ext, flags);

  if (err) {
    throw cuda::Error{err};
  }
  return res;
}

void arr_del(arr_t arr) {
  if (!arr) {
    return;
  }

  if (auto err = ::cudaFreeArray(arr)) {
    throw cuda::Error{err};
  }
}

template <class T>
void arr_set(arr_t arr, const T* src, u32 n, const u32 dims[], const u32 step[], stream_t stream) {
  if (!arr || !src) {
    return;
  }

  if (step[0] != 1) {
    throw cuda::Error{cudaErrorInvalidAddressSpace};
  }

  if (step[1] < dims[1]) {
    throw cuda::Error{cudaErrorInvalidPitchValue};
  }

  const auto ext = cudaExtent{
      n > 0 ? dims[0] : 1,
      n > 1 ? dims[1] : 1,
      n > 2 ? dims[2] : 1,
  };

  const auto src_ptr = cudaPitchedPtr{
      .ptr = src,
      .pitch = step[1] * sizeof(T),
      .xsize = ext.width,
      .ysize = ext.height,
  };

  const auto params = cudaMemcpy3DParms{
      .srcPtr = src_ptr,
      .dstArray = arr,
      .extent = ext,
      .kind = cudaMemcpyDefault,
  };

  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

auto tex_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t {
  if (!arr) {
    return tex_t{0};
  }

  const auto tex_addr_mode = [&] {
    switch (addr_mode) {
      case AddrMode::Border: return cudaAddressModeBorder;
      case AddrMode::Clamp:  return cudaAddressModeClamp;
      default:               return cudaAddressModeBorder;
    }
  }();

  const auto tex_filt_mode = [&] {
    switch (filt_mode) {
      case FiltMode::Point:  return cudaFilterModePoint;
      case FiltMode::Linear: return cudaFilterModeLinear;
      default:               return cudaFilterModePoint;
    }
  }();

  const auto res_desc = cudaResourceDesc{
      .resType = cudaResourceTypeArray,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = cudaTextureDesc{
      .addressMode = {tex_addr_mode, tex_addr_mode, tex_addr_mode},
      .filterMode = tex_filt_mode,
      .readMode = cudaReadModeElementType,
      .normalizedCoords = 0,
  };

  auto tex = cudaTextureObject_t{0};
  if (auto err = cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr)) {
    throw cuda::Error{err};
  }

  return tex;
}

void tex_del(tex_t obj) {
  if (!obj) {
    return;
  }

  if (auto err = cudaDestroyTextureObject(obj)) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda::detail
