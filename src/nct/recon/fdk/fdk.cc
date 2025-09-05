#include "nct/recon/fdk/fdk.h"
#include "nct/recon/fdk/fdk_filt.h"
#include "nct/recon/fdk/fdk_bp.h"

namespace nct::recon {

using namespace math;
using namespace cuda;

FDK::FDK() {}

FDK::~FDK() {}

void FDK::init(const Geo& geo, u32x3 shape, f32x3 pixel) {
  _geo = geo;
  _vol_shape = shape;
  _vol_pixel = pixel;

  this->init_weight();
  this->init_ramp_filter();
  this->init_fft();
}

auto FDK::exec(NdSlice<f32, 3> views) -> cuda::NdArray<f32, 3> {
  this->apply_weight(views);
  this->apply_ramp_filter(views);

  auto vol = NdArray<f32, 3>::with_dim(_vol_shape, MemType::Device);
  this->backward_project(views, *vol);
  return vol;
}

void FDK::init_weight() {
  _weight = _weight.with_dim({_geo.ndet_u, _geo.ndet_v}, cuda::MemType::Managed);

  auto weight = *_weight;
  const auto SAD = _geo.SAD;
  for (auto iv = 0U; iv < _geo.ndet_u; iv++) {
    for (auto iu = 0U; iu < _geo.ndet_v; iu++) {
      const auto [u, v] = _geo.det_pos({iu, iv});
      const auto w = SAD / sqrtf(u * u + v * v + SAD * SAD);
      weight[{iu, iv}] = w;
    }
  }
}

void FDK::init_ramp_filter() {
  const auto full_len = math::up_to_pow2(_geo.ndet_u);
  const auto half_len = full_len / 2 + 1;
  _ramp_filter = _ramp_filter.with_dim({half_len}, cuda::MemType::Managed);

  auto filter = *_ramp_filter;
  for (auto i = 0U; i < half_len; ++i) {
    const auto x = i <= half_len / 2 ? i : half_len - i;
    filter[i] = static_cast<f32>(x) / static_cast<f32>(half_len);
  }
}

void FDK::init_fft() {
  const auto fft_len = math::up_to_pow2(_geo.ndet_u);
  _fft_r2c = cuda::FFT<f32, cf32>::plan_1d({fft_len}, _geo.ndet_v);
  _fft_c2r = cuda::FFT<cf32, f32>::plan_1d({fft_len}, _geo.ndet_v);
}

void FDK::apply_weight(NdSlice<f32, 3> views) {
  _weight.sync_gpu();
  fdk_apply_weight_gpu(views, *_weight);
}

void FDK::apply_ramp_filter(NdSlice<f32, 3> views) {
  _ramp_filter.sync_gpu();

  const auto full_len = math::up_to_pow2(_geo.ndet_u);
  const auto half_len = full_len / 2 + 1;
  auto tmp_r = cuda::NdArray<f32, 2>::with_dim({full_len, _geo.ndet_v});
  auto tmp_c = cuda::NdArray<cf32, 2>::with_dim({half_len, _geo.ndet_v});

  for (auto iview = 0U; iview < views._dims.z; ++iview) {
    auto view = views.slice(iview, ALL{}, ALL{});
    cuda::copy2d(view, *tmp_r);
    _fft_r2c(tmp_r.data(), tmp_c.data());
    fdk_apply_ramp_filter_gpu(views, *_ramp_filter);
    _fft_c2r(tmp_c.data(), tmp_r.data());
    cuda::copy2d(*tmp_r, view);
  }
}

void FDK::backward_project(NdSlice<f32, 3> views, NdSlice<f32, 3> vol) {
  const auto params = FdkBpParams{
      .SAD = _geo.SAD,
      .SDD = _geo.SDD,
      .det_shape = {_geo.ndet_u, _geo.ndet_v},
      .det_pixel = {_geo.det_pixel_u, _geo.det_pixel_v},
      .vol_shape = _vol_shape,
      .vol_pixel = _vol_pixel,
  };

  const auto nview = views._dims.z;

  auto src = cuda::NdArray<f32x3>::with_dim({nview}, MemType::Managed);
  for (auto i = 0u; i < nview; i++) {
    src[{i}] = _geo.src_pos(i);
  }
  src.sync_gpu();

  auto det = cuda::LTexture<f32, 2>::from_slice(views, FiltMode::Linear);
  fdk_bp_gpu(params, *src, *det, vol);
}

}  // namespace nct::recon
