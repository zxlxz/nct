#pragma once

#include "nct/core.h"

namespace nct::cuda {

struct Device {
  struct Info {
    usize memory_size;
    usize sm_count;
  };

 public:
  static auto count() -> int;
  static auto current() -> int;
  static void set(int);
  static auto info(int id) -> Info;
};

}
