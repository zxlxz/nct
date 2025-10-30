#include "nct/correction/det_corr.h"

namespace nct::correction {

DetCorr::DetCorr() noexcept {}

DetCorr::~DetCorr() noexcept {}

void DetCorr::set_dark_tbl(Array<f32, 2> dark) {
  _dark_tbl = mem::move(dark);
}

void DetCorr::set_air_tbl(Array<f32, 2> air) {
  _air_tbl = mem::move(air);
}

void DetCorr::set_beam_harden_tbl(Array<f32, 1> tbl) {
  _beam_harden_tbl = mem::move(tbl);
}

void DetCorr::exec(NView<f32, 3> views) {}

}  // namespace nct::correction
