#pragma once

#include "nct/math.h"

namespace nct::params {

struct ScanParam {
  static constexpr auto TAG = u16{0x1011U};

  i32 raw_data_uid;
  i32 tube_current_ma;
  i32 tube_voltage_kv;
  i32 anode_speed_hz;  // default[50]
  i32 compensator;     // default[0]
  i32 fs_size;         // [FS_SS, FS_LL, FS_SL]
  i32 nffs;            // number of flying focal spots [1, 2]
  i32 diagnostic_level;
  i32 scan_mode;
  i32 manual_auto;
  i32 trigger;
  i32 auto_tilt;
  i32 auto_in_out;
  i32 used_detector_lines;
  i32 slice_width_mm;
  i32 fs_switch_mode;
  i32 fs_switch_sequence;
  i32 fs_size_z_mm;
  i32 wedge_type;
  i32 temperature_reading;
  i32 temperature_samples;
  i32 ncycles;
  f32 spare1[4];
  i32 front_blade_error_slices;
  i32 rear_blade_error_slices;
  f32 front_collimator_to_center_mm;
  f32 readr_collimator_to_center_mm;
  f32 front_collimator_position_mm_per_tick;
  f32 front_collimator_position_offset_mm;
  f32 rear_collimator_position_mm_per_tick;
  f32 rear_collimator_position_offset_mm;
  i32 scan_fee_mode;
  i32 capacitance_pf;
  i32 integration_time_us;
  i32 fee_mode_gain;
  i32 resolution;
  i32 angular_sampling_obsolete;
  f32 tilt_deg;
  f32 swivel_deg;
  i32 start_angle;  // in 0.1 deg
  i32 height_mm;
  f32 start_absolut_bed_position_mm;
  f32 first_delta;
  f32 last_delta;
  f32 surview_rotor_angle_deg;
  i32 nframes_per_scan;
  i32 nrotations_per_scan;
  f32 scan_angle_deg;
  i32 scan_time_msec;
  i32 single_rotation_time_msec;
  i32 rotation_direction;  // [CW=1, CCW=-1]
  f32 bed_speed_mm_sec;
  f32 scan_length_mm;
  i32 direction;  // [IN=-1, OUT=1]
  i32 nscans;
  f32 scan_bed_increment_mm;
  i32 scan_time_increment_us;
  i32 filament_current_ma;
  i32 dont_suspend_recon_after_scan_end;
  i32 nframes_per_series;
  i32 xray_filter_type;
  i32 get_ready_xrt_heat_units;
  i32 get_ready_time;
  i32 dose_modulator;
  i32 xtracking_position_detector;
  char xtracking_position_detector_tab_name[64];
  i32 spare2;
  char detector_position_tab_name[64];
  i32 use_start_angle;
  i32 asymmetric_start_angle;
  i32 res[3];
  f32 dose_factor;
  i32 dom_dose_save_percent;
  i32 dom_modulation_type;
  i32 dom_max_ma;
  f32 colimation_offset_mm;
  f32 total_collimation_mm;
  i32 zdom_start_ma;
  i32 nacquired_energy_levels;
  f32 fs_pos_x_du[8];
  f32 fs_pos_z_du[8];
};

}  // namespace nct::params
