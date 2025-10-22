#pragma once

#include "nct/correction/det_corr_imp.h"
#include "nct/cuda.h"

namespace nct::correction {

using namespace math;
using namespace cuda;

struct Coeffs {
  f32 _ptr[8];
  u32 _len;

 public:
  static auto from(NdSlice<f32, 1> tbl) -> Coeffs {
    if (tbl._dims.x > 8) {
      return {};
    }

    auto res = Coeffs{._len = tbl._dims.x};
    for (auto i = 0U; i < res._len; ++i) {
      res._ptr[i] = tbl[{i}];
    }
    return res;
  }

  __device__ auto operator()(f32 val) const -> f32 {
    auto res = 0.0f;

    auto pow_val = val;
    for (auto i = 0U; i < _len; ++i) {
      res += _ptr[i] * pow_val;
      pow_val *= val;
    }
    return res;
  }
};

__global__ void _det_corr_apply_all_gpu(NdSlice<f32, 3> views,
                                        NdSlice<f32, 2> dark_tbl,
                                        NdSlice<f32, 2> air_tbl,
                                        Coeffs coeffs) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;

  if (iu >= views._dims.x || iv >= views._dims.y) {
    return;
  }

  const auto dark = dark_tbl[{iu, iv}];
  const auto air = air_tbl[{iu, iv}];

  auto ptr = &views[{iu, iv, 0}];
  for (auto iview = 0U; iview < views._dims.z; ++iview) {
    const auto raw = ptr[iview * views._step.z];

    const auto correction = raw - dark;
    const auto norm = correction / (air - dark + 1e-10f);

    const auto p_val = -logf(fmaxf(norm, 1e-10f));
    const auto p_corr = coeffs(p_val);

    ptr[iview * views._step.z] = p_corr;
  }
}

void det_corr_apply_all_gpu(NdSlice<f32, 3> views,
                            NdSlice<f32, 2> dark_tbl,
                            NdSlice<f32, 2> air_tbl,
                            NdSlice<f32, 1> coeffs_tbl) {
  const auto coeffs = Coeffs::from(coeffs_tbl);

  const auto trds = dim3{8, 8, 1};
  const auto blks = cuda::make_blk(views._dims, trds);
  CUDA_RUN(_det_corr_apply_all_gpu, blks, trds)(views, dark_tbl, air_tbl, coeffs);
}

}  // namespace nct::correction
