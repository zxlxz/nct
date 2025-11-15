#pragma once

#include "nct/core.h"
#include "nct/math.h"

namespace nct::data {

struct DMSHead {
  u8 NumOfSlices;             // word 1
  u8 CtType;                  // word 1
  u16 DmsStatus;              // word 2
  u16 DmsError;               // word 3
  u16 IntegrationTime;        // word 4
  u16 ReferenceDetector;      // word 5
  u16 DetectorTemp1;          // word 6
  u16 DetectorTemp2;          // word 7
  u16 DetectorTemp3;          // word 8
  u32 ReadingNumber;          // word 9 & 10
  u16 ASICmodelow;            // word 11
  u16 ASICmodeHigh;           // word 12
  u16 DAS_cnt1;               // word 13
  u16 DAS_cnt2;               // word 14
  u16 ResendCounter;          // word 15
  u16 VerticalTablePosition;  // RA word 16
  i16 TablePosition;          // word 17
  u16 GantryTilt;             // word 18
  u16 RotationAngle;          // word 19
  u16 ReconAngle;             // word 20
  u16 HWlines;                // word 21
  u16 HWlines2;               // word 22
  u32 RcomTimestamp;          // word 23 & 24
  u16 UIDuniqueId;            // word 25
  u16 ShotNumber;             // word 26
  u16 _word27;                // word 27
  u16 AngularPosDenom;        // word 28
  u32 Timestamp;              // word 29 & 30
  u16 DetSamplesPerSlice;     // word 31
  u16 utc_time0;              // word 32
  u16 utc_time1;              // word 33
  u16 utc_time2;              // word 34
  u16 utc_time3;              // word 35
  u16 _word36;                // word 36
  u16 _word37;                // word 37
  u16 DmsType;                // word 38
  u8 CapabilityHigh;          // word 39
  u8 MultiEnergy;             // word 39
  u16 CommandedTubeMa;        // word 40
  u16 XrayTubeMa;             // word 41
  u16 CollimatorBladeRear;    // word 42
  u16 _word43;                // word 43
  u16 _word44;                // word 44
  u16 CollimatorBladeFront;   // word 45
  u16 CapabilityMiddle;       // word 46
  f32 TableExtrapPos;         // word 47 & 48
  u16 _word49;                // word 49
  u16 _word50;                // word 50
  u16 _word51;                // word 51
  u16 _word52;                // word 52
  u16 CapabilityLow;          // word 53
  u16 _word54;                // word 54
  u16 _word55;                // word 55
  u16 _word56;                // word 56
  u32 ReadingNumber2;         // word 57
  u16 SliceNum;               // word 59
  u16 _word60;                // word 60
  u16 _word61;                // word 61
  u16 _word62;                // word 62
  u16 _word63;                // word 63
  u16 _word64;                // word 64

 public:
  auto visit(this auto&& self, auto&& f) {
    f("NumOfSlices", self.NumOfSlices);
    f("CtType", self.CtType);
    f("DmsStatus", self.DmsStatus);
    f("DmsError", self.DmsError);
    f("IntegrationTime", self.IntegrationTime);
    f("ReferenceDetector", self.ReferenceDetector);
    f("DetectorTemp1", self.DetectorTemp1);
    f("DetectorTemp2", self.DetectorTemp2);
    f("DetectorTemp3", self.DetectorTemp3);
    f("ReadingNumber", self.ReadingNumber);
    f("ASICmodelow", self.ASICmodelow);
    f("ASICmodeHigh", self.ASICmodeHigh);
    f("DAS_cnt1", self.DAS_cnt1);
    f("DAS_cnt2", self.DAS_cnt2);
    f("ResendCounter", self.ResendCounter);
    f("VerticalTablePosition", self.VerticalTablePosition);
    f("TablePosition", self.TablePosition);
    f("GantryTilt", self.GantryTilt);
    f("RotationAngle", self.RotationAngle);
    f("ReconAngle", self.ReconAngle);
    f("HWlines", self.HWlines);
    f("HWlines2", self.HWlines2);
    f("RcomTimestamp", self.RcomTimestamp);
    f("UIDuniqueId", self.UIDuniqueId);
    f("ShotNumber", self.ShotNumber);
    f("_word27", self._word27);
    f("AngularPosDenom", self.AngularPosDenom);
    f("Timestamp", self.Timestamp);
    f("DetSamplesPerSlice", self.DetSamplesPerSlice);
    f("utc_time0", self.utc_time0);
    f("utc_time1", self.utc_time1);
    f("utc_time2", self.utc_time2);
    f("utc_time3", self.utc_time3);
    f("_word36", self._word36);
    f("_word37", self._word37);
    f("DmsType", self.DmsType);
    f("CapabilityHigh", self.CapabilityHigh);
    f("MultiEnergy", self.MultiEnergy);
    f("CommandedTubeMa", self.CommandedTubeMa);
    f("XrayTubeMa", self.XrayTubeMa);
    f("CollimatorBladeRear", self.CollimatorBladeRear);
    f("_word43", self._word43);
    f("_word44", self._word44);
    f("CollimatorBladeFront", self.CollimatorBladeFront);
    f("CapabilityMiddle", self.CapabilityMiddle);
    f("TableExtrapPos", self.TableExtrapPos);
    f("_word49", self._word49);
    f("_word50", self._word50);
    f("_word51", self._word51);
    f("_word52", self._word52);
    f("CapabilityLow", self.CapabilityLow);
    f("_word54", self._word54);
    f("_word55", self._word55);
    f("_word56", self._word56);
    f("ReadingNumber2", self.ReadingNumber2);
    f("SliceNum", self.SliceNum);
    f("_word60", self._word60);
    f("_word61", self._word61);
    f("_word62", self._word62);
    f("_word63", self._word63);
    f("_word64", self._word64);
  }

  void fmt(auto& f) const {
    auto imp = f.debug_struct();
    this->visit([&](Str name, const auto& val) {
      if (!name.starts_with("_")) {
        imp.field(name, val);
      }
    });
  }
};

struct DMSFoot {
  u32 readNum;
  u16 sliceNum;
  u16 sliceNumLogic;
  u16 streakCorrection;
  u16 sliceNormCoef2;
  u16 sliceNormCoef1;
  u16 stamp;
};

class RawData {
  u32 _ndet = 0;
  u32 _nslice = 0;
  u32 _view_size = 0u;
  u32 _view_cnt = 0u;
  Vec<u8> _data = {};

 public:
  static auto load_file(Str path) -> RawData;
};

}  // namespace nct::data
