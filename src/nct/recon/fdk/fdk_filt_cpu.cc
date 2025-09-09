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

static auto fdk_ramplak_win(RampWinType win, float r) -> float {
  if (r == 0.0) {
    return 0.0f;
  }
  if (r >= 1.0f) {
    return 0.0f;
  }

  switch (win) {
    case RampWinType::None:       return 1.0f;
    case RampWinType::SheppLogan: return sqrtf(1 - r * r);
    case RampWinType::Cosine:     return cosf(r * PI / 2);
    case RampWinType::Hamming:    return 0.54f + 0.46f * cosf(r * PI);
    case RampWinType::Hann:       return 0.5f + 0.5f * cosf(r * PI);
  }
  return 1.0f;
}

void fdk_init_ramplak_cpu(NdSlice<f32, 1> dst, RampWinType win) {
  static constexpr auto EPSILON = 1e-6f;

  const auto half_len = dst._dims.x;
  const auto full_len = (half_len - 1) * 2;
  const auto denom = static_cast<float>(full_len / 2);

  for (auto i = 0U; i < half_len; ++i) {
    const auto r = f32(i) / denom;
    const auto w = fdk_ramplak_win(win, r);
    dst[{i}] = i == 0 ? EPSILON : r * w;
  }
}

}  // namespace nct::recon
