#include "fdk_imp.h"
#include "nct/cuda/type.h"

namespace nct::recon {

using namespace nct::cuda;

__global__ void _fdk_apply_weight(NdView<f32, 3> views, NdView<f32, 2> weight) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;
  const auto nu = views._size[0];
  const auto nv = views._size[1];
  const auto nk = views._size[2];

  if (iu >= nu || iv >= nv) {
    return;
  }

  const auto w = weight(iu, iv);
  const auto p = &views(iu, iv, 0);
  for (auto k = 0U; k < nk; ++k) {
    p[k * views._step[2]] *= w;
  }
}

__global__ void _fdk_copy_data(NdView<f32, 2> src, NdView<f32, 2> dst) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst._size[0] || y >= dst._size[1]) {
    return;
  }

  auto val = 0.0f;
  if (x < src._size[0] && y < src._size[1]) {
    val = src(x, y);
  }
  dst(x, y) = val;
}

/*!
 * @note filter._size[0] === dst.dims[0]
 */
__global__ void _fdk_apply_filter(NdView<c32, 2> dst, NdView<f32, 1> filter) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst._size[0] || y >= dst._size[1]) {
    return;
  }

  auto k = filter[x];
  auto& c = dst(x, y);
  c.real *= k;
  c.imag *= k;
}

void fdk_apply_weight(NdView<f32, 3> views, NdView<f32, 2> weight) {
  const auto trds = dim3{16, 16};
  CUDA_RUN(_fdk_apply_weight, views._size, trds)(views, weight);
}

void fdk_copy_data(NdView<f32, 2> src, NdView<f32, 2> dst) {
  const auto trds = dim3{16, 16};
  CUDA_RUN(_fdk_copy_data, dst._size, trds)(src, dst);
}

void fdk_mul_filter(NdView<c32, 2> dst, NdView<f32, 1> filter) {
  const auto trds = dim3{16, 16};
  CUDA_RUN(_fdk_apply_filter, dst._size, trds)(dst, filter);
}

}  // namespace nct::recon
