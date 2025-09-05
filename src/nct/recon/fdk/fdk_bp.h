#pragma once

#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/params.h"

namespace nct::recon {

using cuda::LTex;

struct FdkBpParams {
  f32 SAD;          // Source-to-Axis Distance
  f32 SDD;          // Source-to-Detector Distance
  u32x2 det_shape;  // Detector shape
  f32x2 det_pixel;  // Detector pixel size
  u32x3 vol_shape;  // Volume shape
  f32x3 vol_pixel;  // Volume pixel size
};

void fdk_bp_gpu(const FdkBpParams& p, NdSlice<f32x3> src, LTex<f32, 2> det, NdSlice<f32, 3> vol);

}  // namespace nct::recon
