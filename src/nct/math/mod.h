#pragma once

#include "nct/core.h"

namespace nct::math {

static constexpr auto PI = 3.14159265358979323846;

template <class T>
struct alignas(sizeof(T) * 2) complex {
  using Item = T;
  T real;
  T imag;
};

using c32 = complex<f32>;
using c64 = complex<f64>;

}  // namespace nct::math
