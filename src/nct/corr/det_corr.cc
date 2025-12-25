#include "nct/corr/det_corr.h"

namespace nct::corr {

DetCorr::DetCorr() noexcept {}

DetCorr::~DetCorr() noexcept {}

void DetCorr::set_dark_tbl(NdArray<f32, 2> dark) {
  _dark_tbl = mem::move(dark);
}

void DetCorr::set_air_tbl(NdArray<f32, 2> air) {
  _air_tbl = mem::move(air);
}

void DetCorr::set_beam_harden_tbl(NdArray<f32, 1> tbl) {
  _beam_harden_tbl = mem::move(tbl);
}

void DetCorr::exec(NdView<f32, 3> views) {}

}  // namespace nct::corr
