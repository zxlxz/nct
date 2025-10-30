#pragma once

#include "nct/math.h"

namespace nct::recon {

enum class DetType {
  Flat,
  Cylindrical,
  Spherical,
};

struct Params {
  DetType det_type = DetType::Flat;  // detector type

  f32   SOD;             // source->isocenter distance [mm]
  f32   SDD;             // source->detector distance [mm]
  vec2u det_shape;       // number of detector in axial-u [channel]
  vec2f det_pixel;       // detector pixel size in axial-u and axial-v [mm]
  vec3u vol_shape;       // volume shape [voxel]
  vec3f vol_pixel;       // volume pixel size [mm]
  f32   angle_start;     // number of views per revolution
  f32   angle_inc;       // pitch per rev [mm]
  f32   z_start = 0.0f;  // starting angle [rad]
  f32   z_inc = 0.0f;    // starting bed position [mm]

 public:
  __hd__ auto angle(u32 iview) const -> f32 {
    const auto angle_res = angle_start + iview * angle_inc;
    return static_cast<f32>(angle_res);
  }

  __hd__ auto det_pos(u32 iu, u32 iv) const -> vec2f {
    const auto [nu, nv] = this->det_shape;
    const auto [pu, pv] = this->det_pixel;
    const auto u = (iu - (nu - 1.0f) / 2.0f) * pu;
    const auto v = (iv - (nv - 1.0f) / 2.0f) * pv;
    return vec2f{u, v};
  }

  __hd__ auto src(u32 iview) const -> vec3f {
    const auto a = angle_start + static_cast<f32>(iview) * angle_inc;
    const auto z = z_start + static_cast<f32>(iview) * z_inc;
    const auto [cos_a, sin_a] = math::rot(a);
    return vec3f{cos_a * SOD, sin_a * SOD, z};
  }
};

}  // namespace nct::recon
