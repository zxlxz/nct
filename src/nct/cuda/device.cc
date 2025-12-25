#include <cuda_runtime_api.h>
#include "nct/cuda/device.h"
#include "nct/cuda/stream.h"

namespace nct::cuda {

auto Device::count() -> int {
  auto cnt = 0;
  if (auto err = cudaGetDeviceCount(&cnt)) {
    throw Error{err};
  }
  return cnt;
}

auto Device::current() -> int {
  auto id = 0;
  if (auto err = cudaGetDevice(&id)) {
    throw Error{err};
  }
  return id;
}

void Device::set(int id) {
  if (auto err = cudaSetDevice(id)) {
    throw Error{err};
  }
}

auto Device::info(int id) -> Info {
  auto prop = cudaDeviceProp{};
  if (auto err = cudaGetDeviceProperties(&prop, id)) {
    throw Error{err};
  }

  const auto res = Info{
      .memory_size = prop.totalGlobalMem,
      .sm_count = static_cast<u32>(prop.multiProcessorCount),

  };
  return res;
}

}  // namespace nct::cuda
