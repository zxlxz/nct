#pragma once

#include "nct/core.h"

namespace nct::cuda {

using fft_plan_t = int;

auto fft_plan_c2c(const size_t (&dim)[1], size_t batch) -> fft_plan_t;
auto fft_plan_r2c(const size_t (&dim)[1], size_t batch) -> fft_plan_t;
auto fft_plan_c2r(const size_t (&dim)[1], size_t batch) -> fft_plan_t;

void fft_destroy(fft_plan_t plan);

void fft_exec_c2c(fft_plan_t plan, void* in, void* out, int direction);
void fft_exec_r2c(fft_plan_t plan, void* in, void* out);
void fft_exec_c2r(fft_plan_t plan, void* in, void* out);

}  // namespace nct::cuda
