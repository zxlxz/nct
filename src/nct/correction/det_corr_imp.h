#pragma once

#include "nct/math.h"

namespace nct::correction {

using math::NdSlice;

void det_corr_apply_all_gpu(NdSlice<f32, 3> views,
                            NdSlice<f32, 2> dark_tbl,
                            NdSlice<f32, 2> air_tbl,
                            NdSlice<f32, 1> coeffs_tbl);

}  // namespace nct::correction
