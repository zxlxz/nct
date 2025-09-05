#pragma once

#include "nct/math.h"

namespace nct::recon {

using namespace math;

struct Geo {
  f32 SAD;                 // source->isocenter distance [mm]
  f32 SDD;                 // source->detector distance [mm]
  u32 ndet_u;              // number of detector in axial-u [channel]
  u32 ndet_v;              // number of detector in axial-v [row]
  f32 det_pixel_u;         // detector pixel size [mm]
  f32 det_pixel_v;         // detector pixel size [mm]
  u32 nview_per_rev;       // number of views per revolution
  f32 pitch;               // pitch per rev [mm]
  f32 ffs_offset = 0.0f;   // fly focus offset [mm]
  f32 start_angle = 0.0f;  // starting angle [rad]

 public:
  __hd__ auto angle(u32 iview) const -> f32 {
    const auto angle_inc = static_cast<f32>(2 * math::PI / nview_per_rev);
    return static_cast<f32>(iview) * angle_inc + start_angle;
  }

  __hd__ auto src_pos(u32 iview) const -> f32x3 {
    const auto angle = this->angle(iview);
    const auto [cos_a, sin_a] = math::rot(angle);
    const auto off_z = iview * (pitch / nview_per_rev);
    return f32x3{SAD * cos_a, SAD * sin_a, off_z};
  }

  __hd__ auto det_pos(u32x2 idx) const -> f32x2 {
    const auto u = (idx.x - ndet_u / 2 + 0.5f) * det_pixel_u;
    const auto v = (idx.y - ndet_v / 2 + 0.5f) * det_pixel_v;
    return {u, v};
  }
};

}  // namespace nct::recon
