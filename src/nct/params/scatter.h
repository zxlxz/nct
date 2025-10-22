#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

struct ScatterTbl {
  static constexpr auto TAG = u16{0x1065};

  struct {
    i32 scannerType;
    i32 voltage;
    u32 nKxOnDisk = 27;
    u32 npOR = 20;               // [1,50]
    u32 nRx = 41;                // [1,1000]
    u32 nOA = 5;                 // [1,20]
    f32 rXstepMm = 10.0f;        // [1,1000]
    f32 pORstep = 0.05f;         // [0, 1]
    u32 nKx = 167;               // [1, 1000], 167=2*84-1
    u32 nKz = 15;                // [1, 1000], 15=2*8-1
    f32 pixPitchXmm = 11.2664f;  //
    f32 pixPitchZmm = 9.12f;
    i32 spare[4];
  };

  math::NdArray<f32, 1> scale;   // [nKp=npOR*nRx*nOA]
  math::NdArray<u16, 1> index;   // [nKx]
  math::NdArray<f32, 2> weight;  // [4, nKx]
  math::NdArray<f32, 3> kernel;  // [nKp, nKz, nKxOnDisk]

 public:
  void visit(this auto&& self, auto&& f) {
    f("scannerType", self.scannerType);
    f("voltage", self.voltage);
    f("nKxOnDisk", self.nKxOnDisk);
    f("npOR", self.npOR);
    f("nRx", self.nRx);
    f("nOA", self.nOA);
    f("rXstepMm", self.rXstepMm);
    f("pORstep", self.pORstep);
    f("nKx", self.nKx);
    f("nKz", self.nKz);
    f("pixPitchXmm", self.pixPitchXmm);
    f("pixPitchZmm", self.pixPitchZmm);
    f("spare", self.spare);
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
    this->scale = math::NdArray<f32, 1>::with_dim({npOR * nRx * nOA});
    buf.read(this->scale.as_bytes_mut());

    this->index = math::NdArray<u16, 1>::with_dim({nKx});
    buf.read(this->index.as_bytes_mut());

    this->weight = math::NdArray<f32, 2>::with_dim({4, nKx});
    buf.read(this->weight.as_bytes_mut());

    this->kernel = math::NdArray<f32, 3>::with_dim({npOR * nRx * nOA, nKz, nKxOnDisk});
    buf.read(this->kernel.as_bytes_mut());
  }
};

// scatter correction algorithm control parameters
struct ScatterACP {
  static constexpr auto TAG = u16{0x1067};

  struct {
    i32 scannerType;
    i32 nIterations;  // [1,50]
    f32 maxScatFract;
    f32 thickThresLowerMm;
    f32 thickThresUpperMm;
    f32 savGolSizeX;
    f32 savGolSizeZ;
    i32 savGolOrder;
    i32 downSampleFac;
    f32 scatTuningFac;
    i32 nWscatLPX;
    i32 nWscatLPZ;
    f32 wScatScal;
    f32 wScatOffset;
    i32 nCollLines;
    i32 SliceWidth;
    i32 nkV;
    i32 nAcquiredEnergLevels;
    i32 radiusLimCm;
    i32 Spare[13];
  };
};

}  // namespace nct::params
