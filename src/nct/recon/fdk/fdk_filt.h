#pragma once

#include "nct/math.h"
#include "nct/cuda.h"

namespace nct::recon {

using namespace nct::math;

void fdk_apply_weight_gpu(NdSlice<f32, 3> views, NdSlice<f32, 2> weight);
void fdk_apply_ramp_filter_gpu(NdSlice<f32, 3> views, NdSlice<f32, 1> kernel);

}  // namespace nct::recon
