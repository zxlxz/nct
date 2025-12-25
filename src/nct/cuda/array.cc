#include <cuda_runtime_api.h>
#include <channel_descriptor.h>
#include "nct/cuda/array.h"

namespace nct::cuda::detail {

template <class T>
auto arr_new(const size_t (&size)[3], u32 flags) -> arr_t {
  const auto fmt = cudaCreateChannelDesc<T>();
  const auto ext = cudaExtent{size[0], size[1], size[2]};

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
void arr_set(arr_t arr, const T* src, const size_t (&size)[3]) {
  if (arr == nullptr || src == nullptr) {
    return;
  }
  if (size[0] == 0 || size[1] == 0 || size[2] == 0) {
    return;
  }

  const auto ext = cudaExtent{size[0], size[1], size[2]};
  const auto src_ptr = cudaPitchedPtr{
      .ptr = src,
      .pitch = size[0] * sizeof(T),
      .xsize = size[0],
      .ysize = size[1] * size[2],
  };

  const auto params = cudaMemcpy3DParms{
      .srcPtr = src_ptr,
      .dstArray = arr,
      .extent = ext,
      .kind = cudaMemcpyDefault,
  };

  const auto stream = cuda::stream_current();
  const auto err = stream ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params);
  if (err) {
    throw cuda::Error{err};
  }
}

auto tex_new(arr_t arr, int filt_mode, int addr_mode) -> tex_t {
  if (!arr) {
    return tex_t{0};
  }

  const auto res_desc = cudaResourceDesc{
      .resType = cudaResourceTypeArray,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = cudaTextureDesc{
      .addressMode = {addr_mode, addr_mode, addr_mode},
      .filterMode = filt_mode,
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
