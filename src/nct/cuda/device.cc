#include <cuda.h>
#include "nct/cuda/device.h"
#include "nct/cuda/stream.h"
#include "nct/cuda/mem.h"

namespace nct::cuda {

static constexpr auto MAX_DEVICES = 32;

static auto getDeviceId(CUdevice dev) -> int {
  static CUdevice devices[MAX_DEVICES] = {};
  static auto has_init = false;

  if (!has_init) {
    has_init = true;
    const auto device_cnt = Device::count();
    for (auto i = 0; i < device_cnt; ++i) {
      ::cuDeviceGet(&devices[i], i);
    }
  }

  for (auto i = 0; i < MAX_DEVICES; ++i) {
    if (devices[i] == dev) {
      return i;
    }
  }

  return -1;
}

auto Device::count() -> int {
  auto cnt = 0;
  if (auto err = ::cuDeviceGetCount(&cnt)) {
    throw Error{err};
  }
  return cnt;
}

auto Device::get(int id) -> Device {
  auto device = CUdevice{nullptr};
  if (auto err = ::cuDeviceGet(&device, id)) {
    throw Error{err};
  }
  return Device{id, device};
}

auto Device::current() -> Device {
  auto context = CUcontext{nullptr};
  if (auto err = ::cuCtxGetCurrent(&context)) {
    throw Error{err};
  }

  if (context == nullptr) {
    throw Error{CUDA_ERROR_INVALID_CONTEXT};
  }

  auto device = CUdevice{nullptr};
  if (auto err = ::cuCtxGetDevice(&device)) {
    throw Error{err};
  }

  const auto dev_id = getDeviceId(device);
  return Device{dev_id, device};
}

void Device::set_current() {
  // get primary context
  auto context = CUcontext{nullptr};
  if (auto err = ::cuDevicePrimaryCtxRetain(&context, device)) {
    throw Error{err};
  }

  // set context current
  if (auto err = ::cuCtxSetCurrent(context)) {
    throw Error{err};
  }
}

auto Device::name() const -> const char* {
  struct NameInfo {
    CUdevice device = nullptr;
    char name[256] = {};
  };

  static NameInfo _names[MAX_DEVICES] = {};

  auto info_ptr = &_names[0];
  for (auto& info : _names) {
    if (info.device == device) {
      return info.name;
    }
    if (info.device == nullptr) {
      info_ptr = &info;
    }
  }

  if (auto err = ::cuDeviceGetName(info_ptr->name, sizeof(info_ptr->name), device)) {
    throw Error{err};
  }
  return info_ptr->name;
}

auto Device::total_memory() const -> size_t {
  auto bytes = size_t{0};
  if (auto err = ::cuDeviceTotalMem(&bytes, device)) {
    throw Error{err};
  }
  return bytes;
}

}  // namespace nct::cuda
