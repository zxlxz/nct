#include "nct/math.h"
#include "nct/cuda.h"
#include "nct/recon/fdk/fdk_bp.h"

namespace nct::recon {

using namespace cuda;
using namespace math;

namespace {
struct BpParams {
  f32x3 vol_pixel;      // pixel
  f32x3 vol_origin;     // (N/2-0.5)*pixel
  f32x2 det_inv_pixel;  // 1/pixel
  f32x2 det_center;     // N/2-0.5

 public:
  static auto from(const FdkBpParams& p) -> BpParams {
    const auto res = BpParams{
        .vol_pixel = p.vol_pixel,
        .vol_origin = (0.5f * p.vol_shape.as<f32>() + f32x3{0.5, 0.5, 0.5}) * p.vol_pixel,
        .det_inv_pixel = 1.0f / p.det_pixel,
        .det_center = 0.5f * p.det_shape.as<f32>() - f32x2{0.5f, 0.5f},
    };
    return res;
  }

  __device__ auto to_wcs(u32x3 ijk) const -> f32x3 {
    // wcs = ((ijk+0.5) - N/2)*pixel
    //     = ijk*pixel - (N/2-0.5)*pixel
    return ijk.as<f32>() * vol_pixel - vol_origin;
  }

  __device__ auto det_to_pixel(f32x2 uv) const -> f32x2 {
    // uv = ((ij+0.5) - N/2)*pixel
    // ij = uv/pixel + (N/2-0.5)
    return uv * det_inv_pixel + det_center;
  }
};

struct BpTrans {
  f32x3 dir;    // src->det vector [len=1]
  f32x3 vec_u;  // u-axis vector [len=SDD]
  f32x3 vec_v;  // v-axis vector [len=SDD]
  f32 off_u;
  f32 off_v;
  f32 off_d;

 public:
  static auto from(f32 SAD, f32 SDD, f32x3 src) -> BpTrans {
    const auto dir = f32x3{-src.x / SAD, -src.y / SAD, 0.f};
    const auto vec_u = SDD * norm(cross(src, f32x3{0, 0, 1}));
    const auto vec_v = cross(src, vec_u);
    const auto off_u = dot(src, vec_u);
    const auto off_v = dot(src, vec_v);
    const auto off_d = dot(src, dir);
    return {dir, vec_u, vec_v, off_u, off_v, off_d};
  }

  __hd__ auto wcs_to_det(f32x3 wcs) const -> f32x2 {
    const auto scale = 1.0f / (dot(wcs, dir) - off_d);
    const auto num_u = dot(wcs, vec_u) - off_u;
    const auto num_v = dot(wcs, vec_v) - off_v;
    return f32x2{scale * num_u, scale * num_v};
  }
};
}  // namespace

static void _fdk_bp_init_trans(const FdkBpParams& p, NdSlice<f32x3> src, NdSlice<BpTrans> dst) {
  const auto cnt = src._dims.x;
  for (auto i = 0U; i < cnt; ++i) {
    dst[i] = BpTrans::from(p.SAD, p.SDD, src[i]);
  }
}

__global__ void _fdk_bp_gpu(const BpParams params,
                            const BpTrans trans[],
                            NdSlice<f32, 3> vol,
                            LTex<f32, 2> dets) {
  const auto vol_idx = u32x3{
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z,
  };

  if (!vol.in_bounds(vol_idx)) {
    return;
  }

  auto accum = 0.0f;
  const auto vol_wcs = params.to_wcs(vol_idx);
  for (auto proj_idx = 0U; proj_idx < dets._dim.z; ++proj_idx) {
    const auto view = dets[proj_idx];

    const auto det_uv = trans[proj_idx].wcs_to_det(vol_wcs);
    const auto pixel_uv = params.det_to_pixel(det_uv);
    const auto pixel_val = view[pixel_uv];
    accum += pixel_val;
  }

  vol[vol_idx] += accum;
}

void fdk_bp_gpu(const FdkBpParams& p, NdSlice<f32x3> src, LTex<f32, 2> det, NdSlice<f32, 3> vol) {
  const auto gpu_params = BpParams::from(p);

  const auto gpu_trans = cuda::NdArray<BpTrans>::with_dim(src.dims(), MemType::MIXED);
  _fdk_bp_init_trans(p, src, *gpu_trans);

  // bp
  const auto dims = det._dim;
  const auto trds = dim3{8, 8, 4};
  CUDA_RUN(_fdk_bp_gpu, dims, trds)(gpu_params, gpu_trans.data(), vol, det);
}

}  // namespace nct::recon
