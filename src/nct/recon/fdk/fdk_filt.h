#pragma once

#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/params.h"

namespace nct::recon {

using namespace nct::math;

void fdk_init_weight_cpu(NdSlice<f32, 2> dst, const Geo& geo);
void fdk_apply_weight_gpu(NdSlice<f32, 3> dst, NdSlice<f32, 2> val);

void fdk_init_ramp_cpu(NdSlice<f32, 1> val);
void fdk_apply_ramp_gpu(NdSlice<cf32, 2> dst, NdSlice<f32, 1> val);

}  // namespace nct::recon
