#include "nct/cuda/fft.h"
#include <cufft.h>

namespace nct::cuda {

using namespace math;

template <class T>
static auto fft_cast(T* p) -> T* {
  return p;
}

static auto fft_cast(math::cf32* p) -> cufftComplex* {
  return reinterpret_cast<cufftComplex*>(p);
}

static auto fft_cast(math::cf64* p) -> cufftDoubleComplex* {
  return reinterpret_cast<cufftDoubleComplex*>(p);
}

template <class I, class O>
static constexpr auto fft_type() -> cufftType {
  if constexpr (__is_class(I) && __is_class(O)) {
    return sizeof(I) == sizeof(cf32) ? CUFFT_C2C : CUFFT_Z2Z;
  } else if constexpr (__is_class(O)) {
    return sizeof(O) == sizeof(cf32) ? CUFFT_R2C : CUFFT_D2Z;
  } else if constexpr (__is_class(I)) {
    return sizeof(I) == sizeof(cf32) ? CUFFT_C2R : CUFFT_Z2D;
  }
  return CUFFT_C2C;
}

template <class I, class O>
static auto fft_exec(int plan, I* in, O* out, int dir) {
  static constexpr auto type = fft_type<I, O>();
  if constexpr (type == CUFFT_C2C) {
    return cufftExecC2C(plan, fft_cast(in), fft_cast(out), dir);
  } else if constexpr (type == CUFFT_R2C) {
    return cufftExecR2C(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (type == CUFFT_C2R) {
    return cufftExecC2R(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (type == CUFFT_D2Z) {
    return cufftExecD2Z(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (type == CUFFT_Z2D) {
    return cufftExecZ2D(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (type == CUFFT_Z2Z) {
    return cufftExecZ2Z(plan, fft_cast(in), fft_cast(out), dir);
  }
  return CUFFT_SUCCESS;
}

template <class I, class O>
auto FFT<I, O>::plan_1d(const u32 (&dim)[1], u32 batch) -> FFT {
  static constexpr auto type = fft_type<I, O>();
  const auto nx = static_cast<int>(dim[0]);

  auto plan = -1;
  if (auto err = cufftPlan1d(&plan, nx, type, batch)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }

  auto res = FFT{};
  res._plan = plan;
  return res;
}

template <class I, class O>
auto FFT<I, O>::plan_2d(const u32 (&dim)[2], u32 batch) -> FFT {
  static constexpr auto type = fft_type<I, O>();
  int n[2] = {static_cast<int>(dim[0]), static_cast<int>(dim[1])};

  const auto dist = n[0] * n[1];
  const auto half = (n[0] / 2 + 1) * n[1];
  const auto idist = type == CUFFT_C2R || type == CUFFT_Z2D ? half : dist;
  const auto odist = type == CUFFT_R2C || type == CUFFT_D2Z ? half : dist;

  auto plan = -1;
  if (auto err = cufftPlanMany(&plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, type, batch)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }

  auto res = FFT{};
  res._plan = plan;
  return res;
}

template <class I, class O>
void FFT<I, O>::reset() {
  if (_plan == -1) {
    return;
  }
  cufftDestroy(_plan);
  _plan = -1;
}

template <class I, class O>
void FFT<I, O>::operator()(const I* in, O* out, int dir) {
  if (_plan == -1) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }

  if (auto err = fft_exec(_plan, const_cast<I*>(in), out, dir)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

template class FFT<f32, cf32>;
template class FFT<cf32, f32>;
template class FFT<cf32, cf32>;

}  // namespace nct::cuda
