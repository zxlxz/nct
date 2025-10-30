#pragma once

#include "nct/cuda.h"
#include "nct/math.h"

namespace nct::correction {


class DetCorr {
  Array<f32, 2> _dark_tbl;         // [ndet_u, ndet_v]
  Array<f32, 2> _air_tbl;          // [ndet_u, ndet_v]
  Array<f32, 1> _beam_harden_tbl;  // [ndet_u]

 public:
  DetCorr() noexcept;
  ~DetCorr() noexcept;

  void set_dark_tbl(Array<f32, 2> dark);
  void set_air_tbl(Array<f32, 2> air);
  void set_beam_harden_tbl(Array<f32, 1> tbl);

  void exec(NView<f32, 3> views);
};

}  // namespace nct::correction
