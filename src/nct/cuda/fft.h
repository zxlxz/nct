#pragma once

#include "nct/cuda/mod.h"

namespace nct::cuda {

enum {
  FFT_FORWARD = -1,
  FFT_INVERSE = +1,
};

template <class I, class O>
class FFT {
  int _plan{-1};

 public:
  FFT() noexcept {}

  ~FFT() noexcept {
    this->reset();
  }

  FFT(const FFT&) = delete;

  FFT& operator=(const FFT&) = delete;

  FFT(FFT&& other) noexcept {}

  FFT& operator=(FFT&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    _plan = other._plan;
    other._plan = -1;
    return *this;
  }

  static auto plan_1d(const u32 (&dim)[1], u32 batch = 1) -> FFT;
  static auto plan_2d(const u32 (&dim)[2], u32 batch = 1) -> FFT;

  void reset();

  void operator()(const I* in, O* out, int dir = -1);
};

}  // namespace nct::cuda
