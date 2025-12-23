#include "fdk_imp.h"
#include "nct/gpu/fft.h"

namespace nct::recon {

static auto cone_beam_weight(const Params& p, f64 u, f64 v) -> f64 {
  const auto h = p.SOD;
  return h / sqrt(u * u + v * v + h * h);
}

auto fdk_make_weight(const Params& p) -> Array<f32, 2> {
  const auto [nu, nv] = p.det_shape;

  auto res = Array<f32, 2>::with_shape({nu, nv}, MemType::MIXED);

  for (auto iv = 0U; iv < nv; iv++) {
    for (auto iu = 0U; iu < nu; iu++) {
      const auto [u, v] = p.det_pos(iu, iv);
      const auto w = cone_beam_weight(p, u, v);
      res[{iu, iv}] = static_cast<f32>(w);
    }
  }
  return res;
}

static auto ramp_impulse_response(u32 N, f64 T) -> Array<f32, 1> {
  const auto denom = math::PI * math::PI * T * T;

  auto f = [&](f64 s) {
    if (s == 0) {
      return 0.0f;
    }
    return -static_cast<f32>(1.0 / (denom * (s * s)));
  };

  auto res = Array<f32, 1>::with_shape({N}, MemType::MIXED);
  for (u32 i = 0; i < N; i++) {
    const auto s = i < N / 2 ? i : N - i;
    res[i] = f(s);
  }
  return res;
}

auto fdk_make_filter(const Params& p) -> Array<f32, 1> {
  const auto N = gpu::fft_len(p.det_shape.x * 2);
  const auto H = N / 2 + 1;
  const auto T = p.det_pixel.x;

  auto h_data = ramp_impulse_response(N, T);
  auto h_ramp = Array<c32, 1>::with_shape({H}, MemType::MIXED);
  gpu::fft(*h_data, *h_ramp);

  auto h_real = Array<f32, 1>::with_shape({H}, MemType::MIXED);
  for (auto i = 0u; i < H; ++i) {
    const auto r = h_ramp[i].real / N;
    h_real[i] = r;
  }
  h_real.sync_gpu();

  return h_real;
}

void fdk_apply_filter(NView<f32, 3> views, NView<f32, 1> filter) {
  const auto n_det = views._dims[0];
  const auto n_slice = views._dims[1];
  const auto n_proj = views._dims[2];

  const auto pad_len = filter._dims[0];
  const auto fft_len = (pad_len - 1) * 2;

  auto pad_data = Array<f32, 2>::with_shape({pad_len, n_slice}, MemType::GPU);
  auto fft_data = Array<c32, 2>::with_shape({fft_len, n_slice}, MemType::GPU);
  auto pad_view = *pad_data;
  auto fft_view = *fft_data;

  for (auto i_proj = 0U; i_proj < n_proj; ++i_proj) {
    auto view = views.slice_at<2>(i_proj);
    gpu::zero(pad_view);
    gpu::copy(view, pad_view);
    gpu::fft(pad_view, fft_view);
    fdk_mul_filter(fft_view, filter);
    gpu::ifft(fft_view, pad_view);
    gpu::copy(pad_view, view);
  }
}

}  // namespace nct::recon
