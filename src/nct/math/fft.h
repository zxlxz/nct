#pragma once

#include "nct/math/ndview.h"

namespace nct::math {

using u32 = unsigned;
using f32 = float;
using f64 = double;

struct c32 {
  float real;
  float imag;
};

struct c64 {
  double real;
  double imag;
};

auto fft_len(u32 n) -> u32;

template <u32 N>
auto fft(NdView<c32, N> in, NdView<c32, N> out) -> bool;

template <u32 N>
auto ifft(NdView<c32, N> in, NdView<c32, N> out) -> bool;

template <u32 N>
auto fft(NdView<f32, N> in, NdView<c32, N> out) -> bool;

template <u32 N>
auto ifft(NdView<c32, N> in, NdView<f32, N> out) -> bool;

}  // namespace nct::math

namespace nct {
using math::c32;
using math::c64;
}  // namespace nct
