#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

// Hounsfield DetCorr Table
struct HCorTbl {
  static constexpr auto TAG = u16{0x00D9};

  i32 Voltage;
  u32 eResolution;
  i32 nSlice;
  f32 sliceWidthMm;
  i32 nFFS;
  u32 NCoeffs;
  f32 aSpare[10];

 public:
  void visit(this auto&& self, auto&& f) {
    f("voltage", self.Voltage);
    f("eResolution", self.eResolution);
    f("nSlice", self.nSlice);
    f("sliceWidthMm", self.sliceWidthMm);
    f("nFFS", self.nFFS);
    f("nCoeffs", self.NCoeffs);
    f("aSpare", self.aSpare);
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->visit([&](const auto& name, const auto& val) { imp.field(name, val); });
  }

  void load_head(Slice<const u8> buf) {
    this->visit([&](const auto& name, auto& val) { buf.read(mem::as_bytes_mut(val)); });
  }
};

}  // namespace nct::params
