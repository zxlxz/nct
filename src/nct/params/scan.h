#pragma once

#include "nct/params/hdr.h"

namespace nct::params {

struct ScanParam {
  static constexpr auto TAG = u16{0x1011U};
  i32 RawDataUID;
  i32 TubeCurrent_ma;
  i32 TubeVVoltage_kv;
  i32 AnodeSpeed_Hz;  // default[50]
  i32 bCompensator;   // default[0]
  i32 eFsSize;        // [FS_SS, FS_LL, FS_SL]
  i32 nFFS;           // number of flying focal spots [1, 2]
  i32 EDiagnosticLevel;
  i32 eScanMode;
  i32 eManualAuto;
  i32 eTrigger;
  i32 eAutoTilt;
  i32 eAutoInOut;
  i32 nUsedDetectorLines;
  i32 SliceWidth_mm;
  i32 eFsSwitchMode;
  i32 aFsSwitchSequence[MAX_FFS];
  f32 FsSizeX_mm;
  f32 FsSizeZ_mm;
  i32 eWedgeType;
  i32 bTemperatureReading;
  i32 nTemperatureSamples;
  i32 nCycles;
  f32 aSpare1[4];
  i32 FrontBladeErrorSlices;
  i32 RearBladeErrorSlices;
  f32 FrontCollimatorToCenter_mm;
  f32 RearCollimatorToCenter_mm;
  f32 FrontCollimatorPosition_mm_per_tick;
  f32 FrontCollimatorPositionOffset_mm;
  f32 RearCollimatorPosition_mm_per_tick;
  f32 RearCollimatorPositionOffset_mm;
  i32 ScanFeeMode;
  i32 Capacitance_pf;
  i32 IntegrationTime_usec;
  i32 FeeModeGain;
  i32 eResolution;
  i32 eAngularSampling_obsolete;
  f32 Tilt_deg;
  f32 Swivel_deg;
  i32 StartAngle;  // in 0.1 deg
  i32 Height_mm;
  f32 StartAbsolutBedPosition_mm;
  f32 FirstDelta;
  f32 LastDelta;
  f32 SurviewRotorAngle_deg;
  i32 nFramesPerScan;
  i32 nRotationsPerScan;
  f32 ScanAngle_deg;
  i32 ScanTime_msec;
  i32 SingleRotationTime_msec;
  i32 RotationDirection;  // [CW=1, CCW=-1]
  f32 BedSpeed_mm_sec;
  f32 ScanLength_mm;
  i32 eDirection;  // [IN=-1, OUT=1]
  i32 nScans;
  f32 ScanBedIncrement_mm;
  i32 ScanTimeIncrement_microsec;
  i32 FilamentCurrent_ma;
  i32 bDontSuspendReconAfterScanEnd;
  i32 nFramesPerSeries;
  i32 eXRayFilterType;
  i32 GetReadyXrtHeatUnits;
  i32 GetReadyTime;
  i32 bDoseModulator;
  i32 bXTrackingPositionDetector;
  s64 zXTrackingPositionDetectorTabName;
  i32 Spare1;
  s64 DetectorPositionTabName;
  i32 bUseStartAngle;
  i32 bAsymmetricStartAngle;
  i32 aRes[3];
  f32 DoseFactor;
  i32 DomDoseSavePercent;
  i32 DomModulationType;
  i32 DomMax_ma;
  f32 ColimationOffset_mm;
  f32 TotalCollimation_mm;
  i32 ZDOMStart_mA;
  i32 nAcquiredEnergyLevels;
  f32 aFsPosX_du[MAX_FFS];
  f32 aFsPosZ_du[MAX_FFS];

 public:
  void visit(this auto& self, auto&& f) {
    f("RawDataUID", self.RawDataUID);
    f("TubeCurrent_ma", self.TubeCurrent_ma);
    f("TubeVVoltage_kv", self.TubeVVoltage_kv);
    f("AnodeSpeed_Hz", self.AnodeSpeed_Hz);
    f("bCompensator", self.bCompensator);
    f("eFsSize", self.eFsSize);
    f("nFFS", self.nFFS);
    f("EDiagnosticLevel", self.EDiagnosticLevel);
    f("eScanMode", self.eScanMode);
    f("eManualAuto", self.eManualAuto);
    f("eTrigger", self.eTrigger);
    f("eAutoTilt", self.eAutoTilt);
    f("eAutoInOut", self.eAutoInOut);
    f("nUsedDetectorLines", self.nUsedDetectorLines);
    f("SliceWidth_mm", self.SliceWidth_mm);
    f("eFsSwitchMode", self.eFsSwitchMode);
    f("aFsSwitchSequence", self.aFsSwitchSequence);
    f("FsSizeX_mm", self.FsSizeX_mm);
    f("FsSizeZ_mm", self.FsSizeZ_mm);
    f("eWedgeType", self.eWedgeType);
    f("bTemperatureReading", self.bTemperatureReading);
    f("nTemperatureSamples", self.nTemperatureSamples);
    f("nCycles", self.nCycles);
    f("aSpare1", self.aSpare1);
    f("FrontBladeErrorSlices", self.FrontBladeErrorSlices);
    f("RearBladeErrorSlices", self.RearBladeErrorSlices);
    f("FrontCollimatorToCenter_mm", self.FrontCollimatorToCenter_mm);
    f("RearCollimatorToCenter_mm", self.RearCollimatorToCenter_mm);
    f("FrontCollimatorPosition_mm_per_tick", self.FrontCollimatorPosition_mm_per_tick);
    f("FrontCollimatorPositionOffset_mm", self.FrontCollimatorPositionOffset_mm);
    f("RearCollimatorPosition_mm_per_tick", self.RearCollimatorPosition_mm_per_tick);
    f("RearCollimatorPositionOffset_mm", self.RearCollimatorPositionOffset_mm);
    f("ScanFeeMode", self.ScanFeeMode);
    f("Capacitance_pf", self.Capacitance_pf);
    f("IntegrationTime_usec", self.IntegrationTime_usec);
    f("FeeModeGain", self.FeeModeGain);
    f("eResolution", self.eResolution);
    f("eAngularSampling_obsolete", self.eAngularSampling_obsolete);
    f("Tilt_deg", self.Tilt_deg);
    f("Swivel_deg", self.Swivel_deg);
    f("StartAngle", self.StartAngle);
    f("Height_mm", self.Height_mm);
    f("StartAbsolutBedPosition_mm", self.StartAbsolutBedPosition_mm);
    f("FirstDelta", self.FirstDelta);
    f("LastDelta", self.LastDelta);
    f("SurviewRotorAngle_deg", self.SurviewRotorAngle_deg);
    f("nFramesPerScan", self.nFramesPerScan);
    f("nRotationsPerScan", self.nRotationsPerScan);
    f("ScanAngle_deg", self.ScanAngle_deg);
    f("ScanTime_msec", self.ScanTime_msec);
    f("SingleRotationTime_msec", self.SingleRotationTime_msec);
    f("RotationDirection", self.RotationDirection);
    f("BedSpeed_mm_sec", self.BedSpeed_mm_sec);
    f("ScanLength_mm", self.ScanLength_mm);
    f("eDirection", self.eDirection);
    f("nScans", self.nScans);
    f("ScanBedIncrement_mm", self.ScanBedIncrement_mm);
    f("ScanTimeIncrement_microsec", self.ScanTimeIncrement_microsec);
    f("FilamentCurrent_ma", self.FilamentCurrent_ma);
    f("bDontSuspendReconAfterScanEnd", self.bDontSuspendReconAfterScanEnd);
    f("nFramesPerSeries", self.nFramesPerSeries);
    f("eXRayFilterType", self.eXRayFilterType);
    f("GetReadyXrtHeatUnits", self.GetReadyXrtHeatUnits);
    f("GetReadyTime", self.GetReadyTime);
    f("bDoseModulator", self.bDoseModulator);
    f("bXTrackingPositionDetector", self.bXTrackingPositionDetector);
    f("zXTrackingPositionDetectorTabName", self.zXTrackingPositionDetectorTabName);
    f("Spare1", self.Spare1);
    f("DetectorPositionTabName", self.DetectorPositionTabName);
    f("bUseStartAngle", self.bUseStartAngle);
    f("bAsymmetricStartAngle", self.bAsymmetricStartAngle);
    f("aRes", self.aRes);
    f("DoseFactor", self.DoseFactor);
    f("DomDoseSavePercent", self.DomDoseSavePercent);
    f("DomModulationType", self.DomModulationType);
    f("DomMax_ma", self.DomMax_ma);
    f("ColimationOffset_mm", self.ColimationOffset_mm);
    f("TotalCollimation_mm", self.TotalCollimation_mm);
    f("ZDOMStart_mA", self.ZDOMStart_mA);
    f("nAcquiredEnergyLevels", self.nAcquiredEnergyLevels);
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
};

}  // namespace nct::params
