#pragma once

#include "nct/cuda.h"
#include "nct/math.h"

namespace nct::recon {

using namespace math;

class Correct {
  cuda::NdArray<f32, 2> _dark_tbl;         // [ndet_u, ndet_v]
  cuda::NdArray<f32, 2> _air_tbl;          // [ndet_u, ndet_v]
  cuda::NdArray<f32, 1> _beam_harden_tbl;  // [ndet_u]

 public:
  Correct();
  ~Correct();

  void set_dark_tbl(cuda::NdArray<f32, 2> dark);
  void set_air_tbl(cuda::NdArray<f32, 2> air);
  void set_beam_harden_tbl(cuda::NdArray<f32, 1> tbl);

  void exec(NdSlice<f32, 3> views);

 private:
};

}  // namespace nct::recon
