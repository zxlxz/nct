#include "nct/recon/fdk/fdk.h"
#include "nct/recon/fdk/fdk_imp.h"
#include "nct/recon/algos/cone_.h"

namespace nct::recon {

auto FDK::operator()(NView<f32, 3> views) -> Array<f32, 3> {
  auto weight = fdk_make_weight(_params);
  fdk_apply_weight(views, *weight);

  auto filter = fdk_make_filter(_params);
  fdk_apply_filter(views, *filter);

  auto vol = cone_bp(_params, views);
  return vol;
}

}  // namespace nct::recon
