#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/algos/cone_bp.h"

namespace nct::recon {

struct ConeBpGPU {
  f32   SOD;         // source->iso center distance [mm]
  f32   SDD;         // source->detector distance [mm]
  vec3f vol_pixel;   // volume pixel [mm]
  vec3f vol_origin;  // volume origin pos: (N/2-0.5)*pixel [mm]
  vec2f det_pixel;   // detector pixel [mm]
  vec2f det_center;  // detector center N/2-0.5 [px]

 public:
  static auto from(const Params& p) -> ConeBpGPU {
    const auto res = ConeBpGPU{
        .SOD = p.SOD,
        .SDD = p.SDD,
        .vol_pixel = p.vol_pixel,
        .vol_origin = (0.5f * p.vol_shape - vec3f{0.5, 0.5, 0.5}) * p.vol_pixel,
        .det_pixel = p.det_pixel,
        .det_center = 0.5f * p.det_shape - vec2f{0.5f, 0.5f},
    };
    return res;
  }

  // convert: voxel index -> world coordinate system
  __hd__ auto vox_to_world(vec3u vox) const -> vec3f {
    return vox * vol_pixel - vol_origin;
  }

  // convert: detector coordinate -> pixel index
  __hd__ auto det_to_pixel(vec2f loc) const -> vec2f {
    // loc = ((ij+0.5) - N/2)*pixel
    // ij = loc/pixel + (N/2-0.5)
    return loc / det_pixel + det_center;
  }

  // Ray from source s to voxel P, intersecting detector plane
  __hd__ auto ray_intersect_det(vec3f S, vec3f P) const -> vec2f {
    // - rotate -beta around z-axis
    // - translate to source position s
    const auto px = S.x * P.x - S.y * P.y;
    const auto py = S.y * P.x + S.x * P.y;
    const auto pz = P.z - S.z;

    // s = vec3f{SOD, 0, 0};
    // p = vec3f{px, py, pz};
    // r = p - s = {px - SOD, py, pz};
    // t = (SDD - SOD) / r.x
    // q = s + t * r = {SDD, t*py, t*pz}
    const auto r = vec3f{px - SOD, py, pz};
    const auto t = (SDD - SOD) / r.x;

    const auto det = vec2f{-t * py, t * pz};
    return det;
  }
};

__global__ void _cone_bp_gpu(ConeBpGPU params, NView<vec3f> srcs, LTex<f32, 3> views, NView<f32, 3> vol) {
  const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iy = blockIdx.y * blockDim.y + threadIdx.y;
  const auto iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (!vol.in_bounds(ix, iy, iz)) {
    return;
  }

  const auto vox = params.vox_to_world({ix, iy, iz});

  auto accum = 0.0f;
  for (auto proj_idx = 0U; proj_idx < views._dims[2]; ++proj_idx) {
    const auto src = srcs[proj_idx];
    const auto det = params.ray_intersect_det(src, vox);
    const auto pxl = params.det_to_pixel(det);
    if (!views.in_bounds(proj_idx, pxl)) {
      continue;
    }

    const auto val = views(proj_idx, pxl);
    accum += val;
  }

  vol(ix, iy, iz) += accum;
}

static auto make_srcs(const Params& p, u32 nproj) -> Array<vec3f> {
  auto res = Array<vec3f>::with_shape({nproj}, MemType::MIXED);
  for (auto i = 0U; i < nproj; ++i) {
    const auto s = p.src(i);
    res[i] = {s.x / p.SOD, s.y / p.SOD, s.z};
  }
  res.sync_gpu();
  return res;
}

auto cone_bp(const Params& p, NView<f32, 3> views) -> Array<f32, 3> {
  const auto nproj = views._dims[2];

  // prepare data
  auto gpu_params = ConeBpGPU::from(p);

  auto src_array = make_srcs(p, nproj);
  auto views_tex = LTexArr<f32, 3>::from_slice(views);
  auto image_vol = Array<f32, 3>::with_shape(p.vol_shape, MemType::GPU);

  // run
  const auto trds = dim3{8, 8, 8};
  const auto blks = cuda::make_blk(image_vol.dims(), trds);
  CUDA_RUN(_cone_bp_gpu, blks, trds)(gpu_params, *src_array, *views_tex, *image_vol);

  return image_vol;
}

}  // namespace nct::recon
