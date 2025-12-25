#pragma once

#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/params.h"

namespace nct::recon {

auto cone_fp(const Params& p, NdView<f32, 3> vol, u32 nproj) -> NdArray<f32, 3>;
auto cone_bp(const Params& p, NdView<f32, 3> views) -> NdArray<f32, 3>;

}  // namespace nct::recon
