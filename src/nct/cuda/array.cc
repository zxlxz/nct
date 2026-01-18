#include <cuda.h>
#include <type_traits>
#include "nct/cuda/array.h"
#include "nct/cuda/stream.h"

namespace nct::cuda {

template <class T>
static auto get_channel_format() -> CUarray_format {
  if constexpr (std::is_same_v<T, u8>) {
    return CU_AD_FORMAT_UNSIGNED_INT8;
  } else if constexpr (std::is_same_v<T, i8>) {
    return CU_AD_FORMAT_SIGNED_INT8;
  } else if constexpr (std::is_same_v<T, u16>) {
    return CU_AD_FORMAT_UNSIGNED_INT16;
  } else if constexpr (std::is_same_v<T, i16>) {
    return CU_AD_FORMAT_SIGNED_INT16;
  } else if constexpr (std::is_same_v<T, u32>) {
    return CU_AD_FORMAT_UNSIGNED_INT32;
  } else if constexpr (std::is_same_v<T, i32>) {
    return CU_AD_FORMAT_SIGNED_INT32;
  } else if constexpr (std::is_same_v<T, f32>) {
    return CU_AD_FORMAT_FLOAT;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported array element type.");
  }
}

// in bytes
static auto get_format_size(CUarray_format fmt) -> unsigned {
  switch (fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:    return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:   return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:          return 4;
    default:                          {
      throw cuda::Error{CUDA_ERROR_INVALID_VALUE};
    }
  }
}

template <class T>
auto array_new(u32 ndim, const u32 (&size)[3], u32 flags) -> arr_t {
  const auto channel_format = get_channel_format<T>();
  const auto desc = CUDA_ARRAY3D_DESCRIPTOR{
      .Width = size[0],
      .Height = (ndim >= 2) ? size[1] : 1,
      .Depth = (ndim >= 3) ? size[2] : 1,
      .Format = channel_format,
      .NumChannels = 1,
      .Flags = flags,
  };

  auto res = arr_t{nullptr};
  auto err = ::cuArray3DCreate(&res, &desc);

  if (err) {
    throw cuda::Error{err};
  }
  return res;
}

void array_del(arr_t arr) {
  if (!arr) {
    return;
  }

  if (auto err = ::cuArrayDestroy(arr)) {
    throw cuda::Error{err};
  }
}

void array_ext(arr_t arr, u32 (&size)[3]) {
  if (arr == nullptr) {
    return;
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR{};
  if (auto err = ::cuArray3DGetDescriptor(&desc, arr)) {
    throw cuda::Error{err};
  }

  size[0] = static_cast<u32>(desc.Width);
  size[1] = static_cast<u32>(desc.Height);
  size[2] = static_cast<u32>(desc.Depth);
}

void array_set(arr_t arr, const void* src) {
  if (arr == nullptr || src == nullptr) {
    return;
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR{};
  if (auto err = ::cuArray3DGetDescriptor(&desc, arr)) {
    throw cuda::Error{err};
  }

  const auto fmt_size = get_format_size(desc.Format);  // in bytes

  auto copy_params = CUDA_MEMCPY3D{
      .srcXInBytes = 0,  // must be zero
      .srcY = 0,         // must be zero
      .srcZ = 0,         // must be zero
      .srcLOD = 0,       // must be zero
      .srcMemoryType = CU_MEMORYTYPE_UNIFIED,
      .srcHost = src,
      .srcPitch = static_cast<unsigned int>(desc.Width * fmt_size),
      .srcHeight = static_cast<unsigned int>(desc.Height),
      .dstXInBytes = 0,
      .dstY = 0,
      .dstZ = 0,
      .dstLOD = 0,
      .dstMemoryType = CU_MEMORYTYPE_ARRAY,
      .dstArray = arr,
      .WidthInBytes = static_cast<unsigned int>(desc.Width * fmt_size),
      .Height = static_cast<unsigned int>(desc.Height),
      .Depth = static_cast<unsigned int>(desc.Depth),
  };

  const auto stream = cuda::stream_current();
  const auto err = stream ? cuMemcpy3DAsync(&copy_params, stream) : cuMemcpy3D(&copy_params);
  if (err) {
    throw cuda::Error{err};
  }
}

auto texture_new(arr_t arr, FiltMode filt_mode, AddrMode addr_mode) -> tex_t {
  if (!arr) {
    return tex_t{0};
  }

  const auto cu_filt = static_cast<CUfilter_mode>(filt_mode);
  const auto cu_addr = static_cast<CUaddress_mode>(addr_mode);

  const auto res_desc = CUDA_RESOURCE_DESC{
      .resType = CU_RESOURCE_TYPE_ARRAY,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = CUDA_TEXTURE_DESC{
      .addressMode = {cu_addr, cu_addr, cu_addr},
      .filterMode = cu_filt,
      .flags = 0,
  };

  auto tex_obj = tex_t{0};
  if (auto err = cuTexObjectCreate(&tex_obj, &res_desc, &tex_desc, nullptr)) {
    throw cuda::Error{err};
  }
  return tex_obj;
}

void texture_del(tex_t obj) {
  if (!obj) {
    return;
  }

  if (auto err = cuTexObjectDestroy(obj)) {
    throw cuda::Error{err};
  }
}

}  // namespace nct::cuda
