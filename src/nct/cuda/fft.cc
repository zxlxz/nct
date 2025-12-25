#include <cufft.h>

#include "nct/cuda/fft.h"
#include <sfc/collections.h>

namespace nct::cuda {

auto fft_plan_1d(const u32 (&dim)[1], int type, u32 batch) -> fft_plan_t {
  const auto nx = static_cast<int>(dim[0]);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = cufftPlan1d(&plan, nx, type, batch)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
  return plan;
}

void fft_destroy(fft_plan_t plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  if (auto err = cufftDestroy(plan)) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

auto fft_plan_c2c(const u32 (&dim)[1], u32 batch) -> fft_plan_t {
  return fft_plan_1d(dim, CUFFT_C2C, batch);
}

auto fft_plan_r2c(const u32 (&dim)[1], u32 batch) -> fft_plan_t {
  return fft_plan_1d(dim, CUFFT_R2C, batch);
}

auto fft_plan_c2r(const u32 (&dim)[1], u32 batch) -> fft_plan_t {
  return fft_plan_1d(dim, CUFFT_C2R, batch);
}

void fft_exec_c2c(fft_plan_t plan, void* in, void* out, int direction) {
  const auto in_ptr = reinterpret_cast<cufftComplex*>(in);
  const auto out_ptr = reinterpret_cast<cufftComplex*>(out);

  const auto ret = cufftExecC2C(plan, in_ptr, out_ptr, direction);
  if (ret != CUFFT_SUCCESS) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

void fft_exec_r2c(fft_plan_t plan, void* in, void* out) {
  const auto in_ptr = reinterpret_cast<cufftReal*>(in);
  const auto out_ptr = reinterpret_cast<cufftComplex*>(out);

  const auto ret = cufftExecR2C(plan, in_ptr, out_ptr);
  if (ret != CUFFT_SUCCESS) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

void fft_exec_c2r(fft_plan_t plan, void* in, void* out) {
  const auto in_ptr = reinterpret_cast<cufftComplex*>(in);
  const auto out_ptr = reinterpret_cast<cufftReal*>(out);

  const auto ret = cufftExecC2R(plan, in_ptr, out_ptr);
  if (ret != CUFFT_SUCCESS) {
    throw cuda::Error{cudaError_t::cudaErrorInvalidValue};
  }
}

}  // namespace nct::cuda
