#include "fdk_imp.h"
#include "nct/cuda/mod.h"

namespace nct::recon {

__global__ void _fdk_apply_weight(NView<f32, 3> views, NView<f32, 2> weight) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;
  const auto nu = views._dims[0];
  const auto nv = views._dims[1];
  const auto nk = views._dims[2];

  if (iu >= nu || iv >= nv) {
    return;
  }

  const auto w = weight(iu, iv);
  const auto p = &views(iu, iv, 0);
  for (auto k = 0U; k < nk; ++k) {
    p[k * views._step[2]] *= w;
  }
}

__global__ void _fdk_copy_data(NView<f32, 2> src, NView<f32, 2> dst, bool zero_pad) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst._dims[0] || y >= dst._dims[1]) {
    return;
  }

  auto val = 0.0f;
  if (x < src._dims[0] && y < src._dims[1]) {
    val = src(x, y);
  }
  dst(x, y) = val;
}

/*!
 * @note filter._dims[0] === dst.dims[0]
 */
__global__ void _fdk_apply_filter(NView<c32, 2> dst, NView<f32, 1> filter) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst._dims[0] || y >= dst._dims[1]) {
    return;
  }

  auto k = filter[x];
  auto& c = dst(x, y);
  c.real *= k;
  c.imag *= k;
}

void fdk_apply_weight(NView<f32, 3> views, NView<f32, 2> weight) {
  const auto trds = dim3{16, 16};
  const auto blks = cuda::make_blk(views._dims, trds);
  CUDA_RUN(_fdk_apply_weight, blks, trds)(views, weight);
}

void fdk_copy_data(NView<f32, 2> src, NView<f32, 2> dst) {
  const auto trds = dim3{16, 16};
  const auto blks = cuda::make_blk(dst._dims, trds);
  const auto zero_pad = (src._dims[0] < dst._dims[0]);
  CUDA_RUN(_fdk_copy_data, blks, trds)(src, dst, zero_pad);
}

void fdk_mul_filter(NView<c32, 2> dst, NView<f32, 1> filter) {
  const auto trds = dim3{16, 16};
  const auto blks = cuda::make_blk(dst._dims, trds);
  CUDA_RUN(_fdk_apply_filter, blks, trds)(dst, filter);
}

}  // namespace nct::recon
