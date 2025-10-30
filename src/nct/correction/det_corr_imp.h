#pragma once

#include "nct/math.h"

namespace nct::correction {

using math::NView;

void det_corr_apply_all_gpu(NView<f32, 3> views,
                            NView<f32, 2> dark_tbl,
                            NView<f32, 2> air_tbl,
                            NView<f32, 1> coeffs_tbl);

}  // namespace nct::correction
