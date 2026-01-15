#include "fdk_imp.h"
#include "nct/math.h"

namespace nct::recon {

static auto cone_beam_weight(const Params& p, f64 u, f64 v) -> f64 {
  const auto h = p.SOD;
  return h / sqrt(u * u + v * v + h * h);
}

auto fdk_make_weight(const Params& p) -> NdArray<f32, 2> {
  const auto [nu, nv] = p.det_shape;

  auto res = NdArray<f32, 2>::with_shape({nu, nv}, Alloc::UMA);

  auto m = *res;
  for (auto iv = 0U; iv < nv; iv++) {
    for (auto iu = 0U; iu < nu; iu++) {
      const auto [u, v] = p.det_pos(iu, iv);
      const auto w = cone_beam_weight(p, u, v);
      m[iu, iv] = static_cast<f32>(w);
    }
  }
  return res;
}

static auto ramp_impulse_response(u32 N, f64 T) -> NdArray<f32, 1> {
  const auto denom = (M_PI * M_PI) * (T * T);

  auto f = [&](f64 s) {
    if (s == 0) {
      return 0.0f;
    }
    return -static_cast<f32>(1.0 / (denom * (s * s)));
  };

  auto res = NdArray<f32, 1>::with_shape({N}, Alloc::UMA);
  for (u32 i = 0; i < N; i++) {
    const auto s = i < N / 2 ? i : N - i;
    res[i] = f(s);
  }
  return res;
}

auto fdk_make_filter(const Params& p) -> NdArray<f32, 1> {
  const auto N = math::fft_len(p.det_shape[0] * 2);
  const auto H = N / 2 + 1;
  const auto T = p.det_pixel[0];

  auto h_data = ramp_impulse_response(N, T);
  auto h_ramp = NdArray<c32, 1>::with_shape({H}, Alloc::UMA);
  math::fft(*h_data, *h_ramp);

  auto h_real = NdArray<f32, 1>::with_shape({H}, Alloc::UMA);
  for (auto i = 0u; i < H; ++i) {
    const auto r = h_ramp[i].real / N;
    h_real[i] = r;
  }
  h_real.sync_gpu();

  return h_real;
}

void fdk_apply_filter(NdView<f32, 3> views, NdView<f32, 1> filter) {
  const auto n_det = views._size[0];
  const auto n_slice = views._size[1];
  const auto n_proj = views._size[2];

  const auto pad_len = filter._size[0];
  const auto fft_len = (pad_len - 1) * 2;

  auto pad_data = NdArray<f32, 2>::with_shape({pad_len, n_slice}, Alloc::GPU);
  auto fft_data = NdArray<c32, 2>::with_shape({fft_len, n_slice}, Alloc::GPU);
  auto pad_view = *pad_data;
  auto fft_view = *fft_data;

  for (auto i_proj = 0U; i_proj < n_proj; ++i_proj) {
    auto view = views.select(2, i_proj);
    fdk_copy_data(view, pad_view);
    math::fft(pad_view, fft_view);
    fdk_mul_filter(fft_view, filter);
    math::ifft(fft_view, pad_view);
    fdk_copy_data(pad_view, view);
  }
}

}  // namespace nct::recon
