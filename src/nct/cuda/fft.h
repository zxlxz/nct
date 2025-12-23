#pragma once

#include "nct/cuda/mod.h"
#include "nct/math.h"

namespace nct::cuda {

auto fft_len(u32 n) -> u32;

template <u32 N>
void fft(NView<c32, N> in, NView<c32, N> out);

template <u32 N>
void ifft(NView<c32, N> in, NView<c32, N> out);

template <u32 N>
void fft(NView<f32, N> in, NView<c32, N> out);

template <u32 N>
void ifft(NView<c32, N> in, NView<f32, N> out);

}  // namespace nct::cuda
