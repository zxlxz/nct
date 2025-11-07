#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/algos/cone_.h"

namespace nct::recon {

struct ConeFpGPU {
  f32 SOD;           // source->iso center distance [mm]
  f32 SDD;           // source->detector distance [mm]
  vec3f vol_pixel;   // volume pixel [mm]
  vec3f vol_origin;  // volume origin pos: (N/2-0.5)*pixel [mm]
  vec2f det_pixel;   // detector pixel [mm]
  vec2f det_center;  // detector center N/2-0.5 [px]

 public:
  static auto from(const Params& p) -> ConeFpGPU {
    const auto res = ConeFpGPU{
        .SOD = p.SOD,
        .SDD = p.SDD,
        .vol_pixel = p.vol_pixel,
        .vol_origin = (0.5f * p.vol_shape - vec3f{0.5, 0.5, 0.5}) * p.vol_pixel,
        .det_pixel = p.det_pixel,
        .det_center = 0.5f * p.det_shape - vec2f{0.5f, 0.5f},
    };
    return res;
  }

  // get det position in world coordinate system
  __hd__ auto det_to_world(vec2u uv, f32 angle) const -> vec3f {
    return {0, 0, 0};
  }
};

__global__ void _cone_fp_gpu(ConeFpGPU p, NView<vec3f> srcs, Tex<f32, 3> vol, NView<f32, 3> views) {
  const auto iu = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iv = blockIdx.y * blockDim.y + threadIdx.y;
  const auto ip = blockIdx.z * blockDim.z + threadIdx.z;

  if (!views.in_bounds(iu, iv, ip)) {
    return;
  }

  const auto S = srcs[ip];

  // Detector pixel coordinate in world frame
  const auto det_loc = p.det_pixel * (vec2f{static_cast<f32>(iu), static_cast<f32>(iv)} - p.det_center);

  // Compute ray direction in world frame
  // Source is at S = (cos_a*SOD, sin_a*SOD, z_s)
  // Detector point in detector frame: (SDD, u, v) after rotation
  // Need to unrotate to get world coordinates

  const auto cos_a = S.x / p.SOD;
  const auto sin_a = S.y / p.SOD;

  // Detector center in rotated frame: (SDD, 0, 0)
  // Detector pixel in rotated frame: (SDD, -det_loc.x, det_loc.y)
  // Convert back to world frame
  const auto det_x = p.SDD;
  const auto det_y = -det_loc.x;
  const auto det_z = det_loc.y;

  // Inverse rotation: world = R^T * rotated
  const auto det_world_x = cos_a * det_x + sin_a * det_y;
  const auto det_world_y = -sin_a * det_x + cos_a * det_y;
  const auto det_world_z = det_z + S.z;

  const auto det_world = vec3f{det_world_x, det_world_y, det_world_z};

  // Ray direction from source to detector
  const auto ray_diff = det_world - S;
  const auto ray_len = math::len(ray_diff);
  const auto ray_dir = (1.0f / ray_len) * ray_diff;

  // Ray marching through volume with fixed step size
  // For cone beam CT with small cone angle, use XY pixel size (in-plane resolution)
  // Step size can be tuned for quality vs performance trade-off
  // Common multipliers: 1.0 (fast), 0.5 (balanced), 0.25 (high quality)
  const auto min_xy_pixel = ::fminf(p.vol_pixel.x, p.vol_pixel.y);
  const auto ray_step = 1.0f * min_xy_pixel;  // TODO: make multiplier configurable via Params
  const auto max_steps = u32(ray_len / ray_step) + 50U;

  f32 accum = 0.0f;

  for (auto step = 0U; step < max_steps; ++step) {
    const auto world_pos = S + ray_step * static_cast<f32>(step) * ray_dir;

    // Convert world position to volume coordinates
    const auto vol_pos = (world_pos + p.vol_origin) / p.vol_pixel;

    // Check bounds manually
    if (vol_pos.x < 0 || vol_pos.x >= static_cast<f32>(vol._dims[0]) ||
        vol_pos.y < 0 || vol_pos.y >= static_cast<f32>(vol._dims[1]) ||
        vol_pos.z < 0 || vol_pos.z >= static_cast<f32>(vol._dims[2])) {
      continue;
    }

    // Sample volume with texture (hardware interpolation using CUDA Tex3D)
    f32 val = 0.0f;
    tex3D(&val, vol._tex, vol_pos.x, vol_pos.y, vol_pos.z);
    accum += val * ray_step;
  }

  views(iu, iv, ip) = accum;
}

static auto make_srcs(const Params& p, u32 nproj) -> Array<vec3f> {
  auto res = Array<vec3f>::with_shape({nproj}, MemType::MIXED);
  for (auto i = 0U; i < nproj; ++i) {
    const auto s = p.src(i);
    res[i] = s;
  }
  res.sync_gpu();
  return res;
}

auto cone_fp(const Params& p, NView<f32, 3> vol, u32 nproj) -> Array<f32, 3> {
  // prepare data
  auto gpu_params = ConeFpGPU::from(p);

  const auto srcs = make_srcs(p, nproj);

  const auto vol_tex = TexArr<f32, 3>::from_slice(vol, cuda::FiltMode::Linear);

  const auto views_shape = vec3u{p.det_shape.x, p.det_shape.y, nproj};
  auto views = Array<f32, 3>::with_shape(views_shape, MemType::GPU);

  // run
  const auto trds = dim3{8, 8, 8};
  const auto blks = cuda::make_blk<3>(views_shape, trds);
  CUDA_RUN(_cone_fp_gpu, blks, trds)(gpu_params, *srcs, *vol_tex, *views);

  return views;
}

}  // namespace nct::recon
