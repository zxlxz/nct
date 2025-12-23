#include <cufft.h>

#include "nct/cuda/fft.h"
#include <sfc/collections.h>

namespace nct::cuda {

namespace detail {

template <class T>
static auto fft_cast(T* p) {
  if constexpr (__is_same(T, c32)) {
    return reinterpret_cast<cufftComplex*>(p);
  } else if constexpr (__is_same(T, c64)) {
    return reinterpret_cast<cufftDoubleComplex*>(p);
  } else {
    return p;
  }
}

template <class I, class O>
static auto fft_type() -> cufftType {
  if constexpr (__is_same(I, f32) && __is_same(O, c32)) {
    return CUFFT_R2C;
  } else if constexpr (__is_same(I, c32) && __is_same(O, f32)) {
    return CUFFT_C2R;
  } else if constexpr (__is_same(I, c32) && __is_same(O, c32)) {
    return CUFFT_C2C;
  } else if constexpr (__is_same(I, f64) && __is_same(O, c64)) {
    return CUFFT_D2Z;
  } else if constexpr (__is_same(I, c64) && __is_same(O, f64)) {
    return CUFFT_Z2D;
  } else if constexpr (__is_same(I, c64) && __is_same(O, c64)) {
    return CUFFT_Z2Z;
  } else {
    static_assert(false, "nct::cuda::fft_type: Unsupported type combination");
  }
}

template <class I, class O>
static auto fft_plan(const u32 (&dim)[1], u32 batch) -> cufftHandle {
  static const auto type = fft_type<I, O>();

  const auto nx = static_cast<int>(dim[0]);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = cufftPlan1d(&plan, nx, type, batch)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
  return plan;
}

template <class I, class O>
static auto fft_plan(const u32 (&dim)[2], u32 batch) -> cufftHandle {
  static const auto type = fft_type<I, O>();

  const auto nx = static_cast<int>(dim[0]);
  const auto ny = static_cast<int>(dim[1]);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = cufftPlan2d(&plan, nx, ny, type)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
  return plan;
}

static void fft_drop(cufftHandle plan) {
  if (plan == -1) {
    return;
  }
  if (auto err = cufftDestroy(plan)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

template <class I, class O>
static void fft_exec(cufftHandle plan, I* in, O* out, int dir) {
  auto ret = CUFFT_SUCCESS;
  if constexpr (__is_same(I, f32) && __is_same(O, c32)) {
    ret = cufftExecR2C(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (__is_same(I, c32) && __is_same(O, f32)) {
    ret = cufftExecC2R(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (__is_same(I, c32) && __is_same(O, c32)) {
    ret = cufftExecC2C(plan, fft_cast(in), fft_cast(out), dir);
  } else if constexpr (__is_same(I, f64) && __is_same(O, c64)) {
    ret = cufftExecD2Z(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (__is_same(I, c64) && __is_same(O, f64)) {
    ret = cufftExecZ2D(plan, fft_cast(in), fft_cast(out));
  } else if constexpr (__is_same(I, c64) && __is_same(O, c64)) {
    ret = cufftExecZ2Z(plan, fft_cast(in), fft_cast(out), dir);
  }

  if (ret != CUFFT_SUCCESS) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

}  // namespace detail

template <u32 N>
struct PlanInfo {};

auto fft_len(u32 n) -> u32 {
  auto res = 1U;
  while (res < n) {
    res *= 2;
  }

  for (auto x = 1U; x <= n; x *= 2) {
    for (auto y = 1u; y <= n; y *= 3) {
      if (x * y >= n) {
        if (x * y < res) {
          res = x * y;
        }
        break;
      }
    }
  }
  return res;
}

template <class I, class O>
auto fft_plan(const u32 (&dims)[1], u32 batch) -> cufftHandle {
  struct Item {
    cufftHandle plan;
    u32 dims[1];
    u32 batch;
  };

  static auto plans = Vec<Item>{};
  for (const auto& x : plans.as_slice()) {
    if (x.dims[0] == dims[0] && x.batch == batch) {
      return x.plan;
    }
  }

  const auto plan = detail::fft_plan<I, O>(dims, batch);
  plans.push(Item{plan, {dims[0]}, batch});
  return plan;
}

template <u32 N>
void fft(math::NView<c32, N> in, math::NView<c32, N> out) {
  for (auto i = 0U; i < N; ++i) {
    if (in._dims[i] != out._dims[i]) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
  }

  const auto len = in._dims[0];
  const auto batch = in.size() / len;
  const auto plan = fft_plan<c32, c32>({len}, batch);
  detail::fft_exec(plan, in._data, out._data, CUFFT_FORWARD);
}

template <u32 N>
void ifft(math::NView<c32, N> in, math::NView<c32, N> out) {
  for (auto i = 0U; i < N; ++i) {
    if (in._dims[i] != out._dims[i]) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
  }

  const auto len = in._dims[0];
  const auto batch = in.size() / len;
  const auto plan = fft_plan<c32, c32>({len}, batch);
  detail::fft_exec(plan, in._data, out._data, CUFFT_INVERSE);
}

template <u32 N>
void fft(math::NView<f32, N> in, math::NView<c32, N> out) {
  if (in._dims[0] / 2 + 1 != out._dims[0]) {
    throw cuda::Error{cudaErrorInvalidValue};
  }
  for (auto i = 1U; i < N; ++i) {
    if (in._dims[i] != out._dims[i]) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
  }

  const auto len = in._dims[0];
  const auto batch = in.size() / len;
  const auto plan = fft_plan<f32, c32>({len}, batch);
  detail::fft_exec(plan, in._data, out._data, CUFFT_FORWARD);
}

template <u32 N>
void ifft(math::NView<c32, N> in, math::NView<f32, N> out) {
  if (in._dims[0] / 2 + 1 != out._dims[0]) {
    throw cuda::Error{cudaErrorInvalidValue};
  }
  for (auto i = 1U; i < N; ++i) {
    if (in._dims[i] != out._dims[i]) {
      throw cuda::Error{cudaErrorInvalidValue};
    }
  }

  const auto len = in._dims[0];
  const auto batch = in.size() / len;
  const auto plan = fft_plan<c32, f32>({len}, batch);
  detail::fft_exec(plan, in._data, out._data, CUFFT_INVERSE);
}

template void fft(math::NView<c32, 1> in, math::NView<c32, 1> out);
template void fft(math::NView<f32, 1> in, math::NView<c32, 1> out);

template void ifft(math::NView<c32, 1> in, math::NView<c32, 1> out);
template void ifft(math::NView<c32, 1> in, math::NView<f32, 1> out);

}  // namespace nct::cuda
