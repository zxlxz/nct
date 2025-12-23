#pragma once

#include "nct/math.h"
#include "nct/recon/params.h"

namespace nct::recon {

auto fdk_make_weight(const Params& p) -> Array<f32, 2>;
auto fdk_make_filter(const Params& p) -> Array<f32, 1>;

void fdk_apply_weight(NView<f32, 3> dst, NView<f32, 2> val);
void fdk_apply_filter(NView<f32, 3> dst, NView<f32, 1> val);

void fdk_copy_data(NView<f32, 2> src, NView<f32, 2> dst);
void fdk_mul_filter(NView<c32, 2> dst, NView<f32, 1> filter);

}  // namespace nct::recon
