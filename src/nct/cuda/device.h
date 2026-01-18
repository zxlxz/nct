#pragma once

#include "nct/core.h"

struct CUdevice_st;

namespace nct::cuda {

struct Device {
  int id = -1;
  CUdevice_st* device = nullptr;

 public:
  static auto count() -> int;
  static auto get(int id) -> Device;
  static auto current() -> Device;

  void set_current();

  auto name() const -> const char*;
  auto total_memory() const -> size_t;
};

}  // namespace nct::cuda
