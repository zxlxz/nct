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
    const auto cos_a = S.x / SOD;
    const auto sin_a = S.y / SOD;
    const auto px = cos_a * P.x - sin_a * P.y;
    const auto py = sin_a * P.x + cos_a * P.y;
    const auto pz = P.z - S.z;

    // s = vec3f{SOD, 0, 0};
    // p = vec3f{px, py, pz};
    // r = p - s = {px - SOD, py, pz};
    // t = (SDD - SOD) / r.x
    // q = s + t * r = {SDD, t*py, t*pz}
    const auto r = vec3f{px - SOD, py, pz};
    const auto t = (SDD - SOD) / r.x;
    const auto u = -t * py;
    const auto v = t * pz;
    return {u, v};
  }
};

__global__ void _cone_fp_gpu(ConeFpGPU p, NView<vec3f> srcs, Tex<f32, 3> vol, NView<f32, 3> views) {}

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
