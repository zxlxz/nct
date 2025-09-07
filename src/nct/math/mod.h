#pragma once

#include <math.h>
#include "nct/core.h"

#ifndef __CUDACC__
#define __hd__
#else
#define __hd__ __host__ __device__
#endif

namespace nct::math {

static constexpr auto PI = 3.14159265358979323846;

template <class T>
struct alignas(sizeof(T) * 2) Complex {
  T real;
  T imag;
};

using cf32 = Complex<f32>;
using cf64 = Complex<f64>;

#ifndef __CUDACC__
static inline auto rsqrtf(float x) -> float {
  return 1.0f / sqrtf(x);
}
#endif

}  // namespace nct::math
