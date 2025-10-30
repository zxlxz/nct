#pragma once

#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/params.h"

namespace nct::recon {

auto cone_bp(const Params& p, NView<f32, 3> views) -> Array<f32, 3>;

}  // namespace nct::recon
