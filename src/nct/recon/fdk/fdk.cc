#include "nct/recon/fdk/fdk.h"
#include "nct/recon/fdk/fdk_filt.h"
#include "nct/recon/fdk/fdk_bp.h"

namespace nct::recon {

using namespace math;
using namespace cuda;

FDK::FDK(const Geo& geo) : _geo{geo} {}

FDK::~FDK() noexcept {}

void FDK::init(u32x3 shape, f32x3 pixel) {
  _vol_shape = shape;
  _vol_pixel = pixel;

  this->init_weight();
  this->init_filter();
  this->init_fft();
}

auto FDK::exec(NdSlice<f32, 3> views) -> NdArray<f32, 3> {
  auto vol = NdArray<f32, 3>::with_dim(_vol_shape, MemType::GPU);

  this->pre_weight(views);
  this->fdk_filter(views);
  this->cone_bp(views, *vol);
  return vol;
}

auto FDK::fft_len() const -> u32 {
  auto len = 1U;
  while (len < _geo.ndet_u) {
    len *= 2;
  }
  return len;
}

void FDK::init_weight() {
  _weight = NdArray<f32, 2>::with_dim({_geo.ndet_u, _geo.ndet_v}, MemType::MIXED);
  fdk_init_weight_cpu(*_weight, _geo);
}

void FDK::init_filter() {
  const auto full_len = this->fft_len();
  const auto half_len = full_len / 2 + 1;

  _ramplak_win = NdArray<f32, 1>::with_dim({half_len}, MemType::MIXED);
  fdk_init_filter_cpu(*_ramplak_win);
}

void FDK::init_fft() {
  const auto fft_len = this->fft_len();
  _fft_r2c = cuda::FFT<f32, cf32>::plan_1d({fft_len}, _geo.ndet_v);
  _fft_c2r = cuda::FFT<cf32, f32>::plan_1d({fft_len}, _geo.ndet_v);
}

void FDK::pre_weight(NdSlice<f32, 3> views) {
  _weight.sync_gpu();
  fdk_apply_weight_gpu(views, *_weight);
}

void FDK::fdk_filter(NdSlice<f32, 3> views) {
  _ramplak_win.sync_gpu();

  const auto full_len = this->fft_len();
  const auto half_len = full_len / 2 + 1;
  auto tmp_r = NdArray<f32, 2>::with_dim({full_len, _geo.ndet_v});
  auto tmp_c = NdArray<cf32, 2>::with_dim({half_len, _geo.ndet_v});

  for (auto iview = 0U; iview < views._dims.z; ++iview) {
    auto view = views.slice(iview, ALL{}, ALL{});
    cuda::copy2d(view, *tmp_r);
    _fft_r2c(tmp_r.data(), tmp_c.data());
    fdk_apply_filter_gpu(*tmp_c, *_ramplak_win);
    _fft_c2r(tmp_c.data(), tmp_r.data());
    cuda::copy2d(*tmp_r, view);
  }
}

void FDK::cone_bp(NdSlice<f32, 3> views, NdSlice<f32, 3> vol) {
  const auto params = FdkBpParams{
      .SAD = _geo.SAD,
      .SDD = _geo.SDD,
      .det_shape = {_geo.ndet_u, _geo.ndet_v},
      .det_pixel = {_geo.det_pixel_u, _geo.det_pixel_v},
      .vol_shape = _vol_shape,
      .vol_pixel = _vol_pixel,
  };

  const auto nview = views._dims.z;

  auto src = NdArray<f32x3>::with_dim({nview}, MemType::MIXED);
  for (auto i = 0u; i < nview; i++) {
    src[{i}] = _geo.src_pos(i);
  }
  src.sync_gpu();

  auto det = cuda::LTexture<f32, 2>::from_slice(views, FiltMode::Linear);
  fdk_bp_gpu(params, *src, *det, vol);
}

}  // namespace nct::recon
