#pragma once

#include "nct/cuda.h"
#include "nct/math.h"
#include "nct/recon/params.h"

namespace nct::recon {

struct FDK {
  Params _params;

 public:
  auto operator()(NView<f32, 3> views) -> Array<f32, 3>;
};

}  // namespace nct::recon
