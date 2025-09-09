#pragma once

#include "nct/cuda.h"
#include "nct/math.h"
#include "nct/recon/params.h"

namespace nct::recon {

using namespace math;

class FDK {
  Geo _geo;
  u32x3 _vol_shape;
  f32x3 _vol_pixel;

  cuda::FFT<f32, cf32> _fft_r2c{};
  cuda::FFT<cf32, f32> _fft_c2r{};
  NdArray<f32, 2> _weight;       // [ndet_u, ndet_v]
  NdArray<f32, 1> _ramplak_win;  // [fft_len]

 public:
  explicit FDK(const Geo& geo);
  ~FDK() noexcept;

  FDK(const FDK&) = delete;
  FDK& operator=(const FDK&) = delete;

  void init(u32x3 shape, f32x3 pixel);
  auto exec(NdSlice<f32, 3> views) -> NdArray<f32, 3>;

 private:
  auto fft_len() const -> u32;

  void init_weight();
  void init_ramplak();
  void init_fft();

  void apply_weight(NdSlice<f32, 3> views);
  void apply_ramplak(NdSlice<f32, 3> views);
  void backward_project(NdSlice<f32, 3> views, NdSlice<f32, 3> vol);
};

}  // namespace nct::recon
