#pragma once

#include "nct/math.h"
#include "nct/recon/params.h"

namespace nct::recon {

auto fdk_make_weight(const Params& p) -> NdArray<f32, 2>;
auto fdk_make_filter(const Params& p) -> NdArray<f32, 1>;

void fdk_apply_weight(NdView<f32, 3> dst, NdView<f32, 2> val);
void fdk_apply_filter(NdView<f32, 3> dst, NdView<f32, 1> val);

void fdk_copy_data(NdView<f32, 2> src, NdView<f32, 2> dst);
void fdk_mul_filter(NdView<c32, 2> dst, NdView<f32, 1> filter);

}  // namespace nct::recon
