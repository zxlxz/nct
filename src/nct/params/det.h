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
    i32 sliceWidth_um;
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
  math::Array<f32, 3> xpos = {};  // [nDetector][nSlice][nFFS]
  math::Array<f32, 3> ypos = {};  // [nDetector][nSlice][nFFS]
  math::Array<f32, 3> zpos = {};  // [nDetector][nSlice][nFFS]

 public:
  void visit(this auto&& self, auto&& f) {
    f("DMSType", self.DMSType);
    f("nFFS", self.nFFS);
    f("nSlices", self.nSlices);
    f("nDetectors", self.nDetectors);
    f("sliceWidth_um", self.sliceWidth_um);
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
    this->visit([&](const auto& name, auto& val) { (void)buf.read(mem::as_bytes_mut(val)); });
  }

  void load_data(Slice<const u8> buf) {
    this->xpos = math::Array<f32, 3>::with_shape({nDetectors, nSlices, nFFS});
    this->ypos = math::Array<f32, 3>::with_shape({nDetectors, nSlices, nFFS});
    this->zpos = math::Array<f32, 3>::with_shape({nDetectors, nSlices, nFFS});
    (void)buf.read(xpos.as_bytes_mut());
    (void)buf.read(ypos.as_bytes_mut());
    (void)buf.read(zpos.as_bytes_mut());
  }

  void save_data(auto& out) const {
    out.write(xpos.as_bytes());
    out.write(ypos.as_bytes());
    out.write(zpos.as_bytes());
  }

};

}  // namespace nct::params
