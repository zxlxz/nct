#pragma once

#include "nct/cuda.h"
#include "nct/math.h"

namespace nct::correct {

using namespace math;

class Correct {
  NdArray<f32, 2> _dark_tbl;         // [ndet_u, ndet_v]
  NdArray<f32, 2> _air_tbl;          // [ndet_u, ndet_v]
  NdArray<f32, 1> _beam_harden_tbl;  // [ndet_u]

 public:
  Correct() noexcept;
  ~Correct() noexcept;

  void set_dark_tbl(NdArray<f32, 2> dark);
  void set_air_tbl(NdArray<f32, 2> air);
  void set_beam_harden_tbl(NdArray<f32, 1> tbl);

  void exec(NdSlice<f32, 3> views);
};

}  // namespace nct::correct
