#pragma once

#include "nct/corr/det_corr_imp.h"
#include "nct/cuda/type.cuh"

namespace nct::corr {

using namespace nct::cuda;

struct Coeffs {
  f32 _ptr[8];
  u32 _len;

 public:
  static auto from(NdView<f32> tbl) -> Coeffs {
    if (tbl._size[0] > 8) {
      return {};
    }

    auto res = Coeffs{._len = tbl._size[0]};
    for (auto i = 0U; i < res._len; ++i) {
      res._ptr[i] = tbl[i];
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

__global__ void _det_corr_apply_all_gpu(NdView<f32, 3> views,
                                        NdView<f32, 2> dark_tbl,
                                        NdView<f32, 2> air_tbl,
                                        Coeffs coeffs) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;
  const auto nu = views._size[0];
  const auto nv = views._size[1];
  const auto nw = views._size[2];

  if (iu >= nu || iv >= nv) {
    return;
  }

  const auto dark = dark_tbl.get(iu, iv);
  const auto air = air_tbl.get(iu, iv);

  auto ptr = &views[{iu, iv, 0}];

  for (auto iw = 0U; iw < nw; ++iw) {
    const auto orig = ptr[iw * views._step[2]];
    const auto corr = orig - dark;
    const auto norm = corr / (air - dark + 1e-10f);
    const auto p_val = -logf(fmaxf(norm, 1e-10f));
    const auto p_corr = coeffs(p_val);
    ptr[iw * views._step[2]] = p_corr;
  }
}

void det_corr_apply_all_gpu(NdView<f32, 3> views,
                            NdView<f32, 2> dark_tbl,
                            NdView<f32, 2> air_tbl,
                            NdView<f32, 1> coeffs_tbl) {
  const auto coeffs = Coeffs::from(coeffs_tbl);

  const auto trds = cuda::dim3{8, 8, 1};
  CUDA_RUN(_det_corr_apply_all_gpu, views._size, trds)(views, dark_tbl, air_tbl, coeffs);
}

}  // namespace nct::corr
