#include "fdk_filt.h"

namespace nct::recon {

using namespace math;
using namespace cuda;

__global__ void _fdk_apply_weight_gpu(NdSlice<f32, 3> views, NdSlice<f32, 2> weight) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;

  if (iu >= views._dims.x || iv >= views._dims.y) {
    return;
  }

  const auto w = weight[{iu, iv}];
  const auto p = &views[{iu, iv, 0}];
  for (auto k = 0U; k < views._dims.z; ++k) {
    p[k * views._step.z] *= w;
  }
}

__global__ void _fdk_apply_ramp_gpu(NdSlice<cf32, 2> fft_c, NdSlice<f32, 1> window) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;

  if (iu >= fft_c._dims.x || iv >= fft_c._dims.y) {
    return;
  }

  const auto k = window[{iu}];
  auto& c = fft_c[{iu, iv}];
  c.real *= k;
  c.imag *= k;
}

void fdk_apply_weight_gpu(NdSlice<f32, 3> views, NdSlice<f32, 2> weight) {
  CUDA_RUN(_fdk_apply_weight_gpu, views._dims, dim3(16, 16))(views, weight);
}

void fdk_apply_ramp_gpu(NdSlice<cf32, 2> fft_c, NdSlice<f32, 1> kernel) {
  CUDA_RUN(_fdk_apply_ramp_gpu, fft_c._dims, dim3(16, 16))(fft_c, kernel);
}

}  // namespace nct::recon
