#pragma once

#include "nct/math.h"

namespace nct::params {

// Hounsfield DetCorr Table
struct HCorTbl {
  static constexpr auto TAG = u16{0x00D9};

  i32 voltage;
  u32 resolution;
  i32 nslice;
  f32 slice_width_mm;
  i32 nffs;
  u32 ncoeffs;
  f32 coeffs[10];

 public:
  void map(this auto&& self, auto&& f) {
    f("voltage", self.voltage);
    f("resolution", self.resolution);
    f("nslice", self.nslice);
    f("slice_width_mm", self.slice_width_mm);
    f("nffs", self.nffs);
    f("ncoeffs", self.ncoeffs);
    f("coeffs", Slice{self.coeffs, self.ncoeffs});
  }

  static auto from_raw(Slice<const u8> buf) -> HCorTbl {
    auto reader = io::Read{buf};

    auto res = HCorTbl{};
    res.map([&](const auto& _, auto&& val) {
      if constexpr (requires { val.as_bytes_mut(); }) {
        reader.read(val.as_bytes_mut());
      } else {
        reader.read_raw(val);
      }
    });
    return res;
  }

  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->map([&](const auto& name, const auto& val) { imp.field(name, val); });
  }
};

}  // namespace nct::params
