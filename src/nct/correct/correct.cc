#include "nct/correct/correct.h"

namespace nct::correct {

Correct::Correct() noexcept {}

Correct::~Correct() noexcept {}

void Correct::set_dark_tbl(NdArray<f32, 2> dark) {
  _dark_tbl = mem::move(dark);
}

void Correct::set_air_tbl(NdArray<f32, 2> air) {
  _air_tbl = mem::move(air);
}

void Correct::set_beam_harden_tbl(NdArray<f32, 1> tbl) {
  _beam_harden_tbl = mem::move(tbl);
}

void Correct::exec(NdSlice<f32, 3> views) {}

}  // namespace nct::correct
