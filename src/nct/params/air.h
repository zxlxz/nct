#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

struct AirCorTbl {
  static constexpr auto TAG = u16{0x1050};

  struct {
    i32 DMSType;
    i32 scannerType;
    i32 nDmsModules;
    i32 AirTypes;
    u32 nSectors;
    u32 nSlices;
    u32 nFFS;
    u32 nDetectors;
    u32 nFanOffset;
    i32 SliceWidth;  // unit: mm
    i32 eResolution;
    i32 Voltage;
    i32 SingleRotationTime;  // unit: us
    i32 Current;             // x-ray tube current 20 to 600 ma
    f32 Temperature;
    i32 WedgeType;
    i32 FilterType;
    i32 EFsSize;  // [SS, LL, SL]
    f32 aFsPosX_du[MAX_FFS];
    f32 aFsPosZ_du[MAX_FFS];
    s64 OriginTabName1 = {};
    s64 OriginTabName2 = {};
    s64 OriginTabName3 = {};
    i32 bCrossTalkCorr;
    f32 rightEdgeAvgXT;
    f32 leftEdgeAvgXT;
    i32 bRefDetectorCalibrated;
    i32 eIrsSubStages;
    i32 eIrsSubStages2;
    f32 AvgRefDetUnlog;
    u32 Spare[7];
  };

  math::Array<f32, 3> data = {};  // [nFFS][nSlice][nDetector+2]

 public:
  void visit(this auto& self, auto&& f) {
    f("DMSType", self.DMSType);
    f("scannerType", self.scannerType);
    f("nDmsModules", self.nDmsModules);
    f("AirTypes", self.AirTypes);
    f("nSectors", self.nSectors);
    f("nSlices", self.nSlices);
    f("nFFS", self.nFFS);
    f("nDetectors", self.nDetectors);
    f("nFanOffset", self.nFanOffset);
    f("SliceWidth", self.SliceWidth);
    f("eResolution", self.eResolution);
    f("Voltage", self.Voltage);
    f("SingleRotationTime", self.SingleRotationTime);
    f("Current", self.Current);
    f("Temperature", self.Temperature);
    f("WedgeType", self.WedgeType);
    f("FilterType", self.FilterType);
    f("EFsSize", self.EFsSize);
    f("aFsPosX_du", self.aFsPosX_du);
    f("aFsPosZ_du", self.aFsPosZ_du);
    f("OriginTabName1", self.OriginTabName1);
    f("OriginTabName2", self.OriginTabName2);
    f("OriginTabName3", self.OriginTabName3);
    f("bCrossTalkCorr", self.bCrossTalkCorr);
    f("rightEdgeAvgXT", self.rightEdgeAvgXT);
    f("leftEdgeAvgXT", self.leftEdgeAvgXT);
    f("bRefDetectorCalibrated", self.bRefDetectorCalibrated);
    f("eIrsSubStages", self.eIrsSubStages);
    f("eIrsSubStages2", self.eIrsSubStages2);
    f("AvgRefDetUnlog", self.AvgRefDetUnlog);
  }

  // trait: fmt::Display
  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->visit([&](const auto& name, const auto& val) { imp.field(name, val); });
  }

  // trait: serde::Deserialize
  void load_head(Slice<const u8> buf) {
    this->visit([&](const auto& name, auto& val) { (void)buf.read(mem::as_bytes_mut(val)); });
  }

  void load_data(Slice<const u8> buf) {
    this->data = math::Array<f32, 3>::with_shape({nFFS, nSlices, nDetectors + 2});
    (void)buf.read(this->data.as_bytes_mut());
  }
};

}  // namespace nct::params
