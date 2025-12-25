#pragma once

#include "nct/math.h"

namespace nct::corr {

using math::NdView;

void det_corr_apply_all_gpu(NdView<f32, 3> views,
                            NdView<f32, 2> dark_tbl,
                            NdView<f32, 2> air_tbl,
                            NdView<f32, 1> coeffs_tbl);

}  // namespace nct::corr
