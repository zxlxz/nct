#pragma once

#include "nct/correct/correct_impl.h"

namespace nct::recon {

using namespace math;
using namespace cuda;

template <class T, u32 N>
struct FixedArr {
  u32 len;
  T ptr[N];

  auto operator[](u32 i) const -> T {
    return ptr[i];
  }
};

__global__ void _corr_apply_elements(NdSlice<f32, 3> views,
                                     NdSlice<f32, 2> dark_tbl,
                                     NdSlice<f32, 2> air_tbl,
                                     FixedArr<f32, 8> coeffs) {
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

    const auto corr = raw - dark;
    const auto norm = corr / (air - dark + 1e-10f);
    const auto pval = -logf(fmaxf(norm, 1e-10f));

    auto p_corr = 0.0f;
    {
      auto pow_p = pval;
      for (auto coeff_idx = 0U; coeff_idx < coeffs.len; ++coeff_idx) {
        p_corr += coeffs[coeff_idx] * pow_p;
        pow_p *= pval;
      }
    }

    ptr[iview * views._step.z] = p_corr;
  }
}

}  // namespace nct::recon
