#pragma once

#include "nct/math/ndview.h"

namespace nct::math {

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

auto fft_len(usize n) -> usize;

template <usize N>
auto fft(NdView<c32, N> in, NdView<c32, N> out) -> bool;

template <usize N>
auto ifft(NdView<c32, N> in, NdView<c32, N> out) -> bool;

template <usize N>
auto fft(NdView<f32, N> in, NdView<c32, N> out) -> bool;

template <usize N>
auto ifft(NdView<c32, N> in, NdView<f32, N> out) -> bool;

}  // namespace nct::math

namespace nct {
using math::c32;
using math::c64;
}  // namespace nct
