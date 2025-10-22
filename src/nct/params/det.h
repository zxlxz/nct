#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

struct DetPosTbl {
  static constexpr auto TAG = u16{0x108D};

  struct {
    i32 DMSType;
    u32 nFFS;
    u32 nSlices;
    u32 nDetectors;
    i32 sliceWidthUm;
    u32 nSectors;
    i32 eResolution;
    i32 Voltage;
    i32 SingleRotationTime;
    i32 Current;  // current tube mA
    f32 Temperature;
    i32 eWedgeType;
    i32 FilterType;
    i32 EFsSize;  // [SS, LL, SL]
    f32 aFsPosX_du[MAX_FFS];
    f32 aFsPosZ_du[MAX_FFS];
  };
  math::NdArray<f32, 3> xpos = {};  // [nFFS][nSlice][nDetector]
  math::NdArray<f32, 3> ypos = {};  // [nFFS][nSlice][nDetector]
  math::NdArray<f32, 3> zpos = {};  // [nFFS][nSlice][nDetector]

 public:
  void visit(this auto&& self, auto&& f) {
    f("DMSType", self.DMSType);
    f("nFFS", self.nFFS);
    f("nSlices", self.nSlices);
    f("nDetectors", self.nDetectors);
    f("sliceWidthUm", self.sliceWidthUm);
    f("nSectors", self.nSectors);
    f("eResolution", self.eResolution);
    f("Voltage", self.Voltage);
    f("SingleRotationTime", self.SingleRotationTime);
    f("Current", self.Current);
    f("Temperature", self.Temperature);
    f("eWedgeType", self.eWedgeType);
    f("FilterType", self.FilterType);
    f("EFsSize", self.EFsSize);
    f("aFsPosX_du", self.aFsPosX_du);
    f("aFsPosZ_du", self.aFsPosZ_du);
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->visit([&](const auto& name, const auto& val) { imp.field(name, val); });
  }

  void load_head(Slice<const u8> buf) {
    this->visit([&](const auto& name, auto& val) { buf.read(mem::as_bytes_mut(val)); });
  }

  void load_data(Slice<const u8> buf) {
    this->xpos = math::NdArray<f32, 3>::with_dim({nFFS, nSlices, nDetectors});
    this->ypos = math::NdArray<f32, 3>::with_dim({nFFS, nSlices, nDetectors});
    this->zpos = math::NdArray<f32, 3>::with_dim({nFFS, nSlices, nDetectors});
    buf.read(xpos.as_bytes_mut());
    buf.read(ypos.as_bytes_mut());
    buf.read(zpos.as_bytes_mut());
  }
};

}  // namespace nct::params
