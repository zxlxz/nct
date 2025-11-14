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
  u16 Res11;                  // word 27
  u16 AngularPosDenom;        // word 28
  u32 Timestamp;              // word 29 & 30
  u16 DetSamplesPerSlice;     // word 31
  u16 UtcTime1;               // word 32 Least significant bit = 10 usec
  u16 UtcTime2;               // word 33 Least significant bit = 10 usec
  u16 UtcTime3;               // word 34 Least significant bit = 10 usec
  u16 UtcTime4;               // word 35 Least significant bit = 10 usec
  u16 Res6[2];                // word 36-37
  u8 DmsType;                 // word 38
  u8 Res15;                   // word 38
  u8 CapabilityHigh;          // word 39
  u8 MultiEnergy;             // word 39
  u16 CommandedTubeMa;        // word 40
  u16 XrayTubeMa;             // word 41
  u16 CollimatorBladeRear;    // word 42
  u16 word43;                 // word 43
  u16 word44;                 // word 44
  u16 CollimatorBladeFront;   // word 45
  u16 CapabilityMiddle;       // word 46
  f32 TableExtrapPos;         // word 47 & 48
  u16 word49;                 // word 49
  u16 word50;                 // word 50
  u16 word51;                 // word 51
  u16 word52;                 // word 52
  u16 CapabilityLow;          // word 53
  u16 word54;                 // word 54
  u16 word55;                 // word 55
  u16 word56;                 // word 56
  u32 ReadingNumber2;         // word 57
  u16 SliceNum;               // word 59
  u16 word60;                 // word 60
  u16 word61;                 // word 61
  u16 word62;                 // word 62
  u16 word63;                 // word 63
  u16 word64;                 // word 64
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
