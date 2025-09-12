#pragma once

#include "nct/math.h"

namespace nct::params {

struct DetPosTbl {
  static constexpr auto TAG = u16{0x108DU};

  i32 dms_type;
  i32 nffs;
  i32 nslices;
  i32 ndetectors;
  i32 slice_width_mm;
  i32 nsectors;
  i32 resolution;
  i32 voltage;
  i32 single_rotation_time;
  i32 curret_tube_ma;
  f32 temperature;
  i32 wedge_type;
  i32 filter_type;
  i32 fs_size;  // [SS, LL, SL]
  f32 fs_pos_x_du[8];
  f32 fs_pos_z_du[8];

  math::NdArray<f32, 3> xpos;
  math::NdArray<f32, 3> ypos;
  math::NdArray<f32, 3> zpos;
};

}  // namespace nct::params
