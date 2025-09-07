#include "fdk_filt.h"

namespace nct::recon {

using namespace math;

void fdk_init_weight_cpu(NdSlice<f32, 2> dst, const Geo& geo) {
  const auto SAD = geo.SAD;
  const auto nu = dst._dims.x;
  const auto nv = dst._dims.y;

  for (auto iv = 0U; iv < nv; iv++) {
    for (auto iu = 0U; iu < nu; iu++) {
      const auto [u, v] = geo.det_pos({iu, iv});
      const auto w = SAD / sqrtf(u * u + v * v + SAD * SAD);
      dst[{iu, iv}] = w;
    }
  }
}

void fdk_init_ramp_cpu(NdSlice<f32, 1> dst) {
  const auto half_len = dst._dims.x;
  const auto full_len = (half_len - 1) * 2;

  for (auto i = 0U; i < half_len; ++i) {
    const auto val = f32(i) / f32(full_len / 2);
    dst[i] = val == 0.f ? 1e-10f : val;
  }
}

}  // namespace nct::recon
