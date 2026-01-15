#include "nct/math/fft.h"
#include "nct/cuda/fft.h"

namespace nct::math {

using cuda::fft_plan_t;

class FFT {
  fft_plan_t _plan = -1;

 public:
  explicit FFT(fft_plan_t plan) : _plan{plan} {}

  ~FFT() {
    if (_plan != -1) {
      cuda::fft_destroy(_plan);
    }
  }

  FFT(FFT&& other) noexcept : _plan{mem::move(other._plan)} {
    other._plan = -1;
  }

  FFT& operator=(FFT&& other) noexcept {
    if (this != &other) {
      mem::swap(_plan, other._plan);
    }
    return *this;
  }

  template <class I, class O>
  static auto xnew(const u32 (&dims)[1], u32 batch) -> FFT {
    if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
      return FFT{cuda::fft_plan_c2c(dims, batch)};
    } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
      return FFT{cuda::fft_plan_r2c(dims, batch)};
    } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
      return FFT{cuda::fft_plan_c2r(dims, batch)};
    } else {
      static_assert(false, "fft_plan: unsupported type combination");
    }
  }

  template <class I, class O>
  void exec(I* in, O* out, int direction = 0) {
    (void)direction;
    if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
      cuda::fft_exec_c2c(_plan, in, out, direction);
    } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
      cuda::fft_exec_r2c(_plan, in, out);
    } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
      cuda::fft_exec_c2r(_plan, in, out);
    } else {
      static_assert(false, "fft_exec: unsupported type combination");
    }
  }
};

template <class Key, class Val>
class KVCache {
  struct Item {
    Key key;
    Val val;
  };
  Vec<Item> _buff{};

 public:
  auto get_or(const auto& key, auto&& f) -> Val& {
    // search
    for (auto& x : _buff.as_mut_slice()) {
      if (x.key == key) {
        return x.val;
      }
    }

    _buff.push(Item{key, f()});
    return _buff.last_mut()->val;
  }
};

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
auto fft_plan(const u32 (&dims)[1], u32 batch) -> FFT& {
  struct Key {
    u32 dims[1];
    u32 batch;

    auto operator==(const Key& other) const -> bool {
      return dims[0] == other.dims[0] && batch == other.batch;
    }
  };

  using Cache = KVCache<Key, FFT>;
  static auto cache = Cache{};

  auto& val = cache.get_or(Key{dims[0], batch}, [&]() { return FFT::xnew<I, O>(dims, batch); });
  return val;
}

template <u32 N>
auto fft(NdView<c32, N> in, NdView<c32, N> out) -> bool {
  // check dims
  for (auto i = 0U; i < N; ++i) {
    if (in._size[i] != out._size[i]) {
      return false;
    }
  }

  const auto len = in._size[0];
  const auto batch = in.numel() / len;

  auto& plan = fft_plan<c32, c32>({len}, batch);
  plan.exec(in._data, out._data, -1);

  return true;
}

template <u32 N>
auto ifft(NdView<c32, N> in, NdView<c32, N> out) -> bool {
  // check dims
  for (auto i = 0U; i < N; ++i) {
    if (in._size[i] != out._size[i]) {
      return false;
    }
  }

  const auto len = in._size[0];
  const auto batch = in.numel() / len;
  auto& plan = fft_plan<c32, c32>({len}, batch);
  plan.exec(in._data, out._data, 1);
  return true;
}

template <u32 N>
auto fft(math::NdView<f32, N> in, math::NdView<c32, N> out) -> bool {
  // check dims
  if (in._size[0] / 2 + 1 != out._size[0]) {
    return false;
  }
  for (auto i = 1U; i < N; ++i) {
    if (in._size[i] != out._size[i]) {
      return false;
    }
  }

  const auto len = in._size[0];
  const auto batch = in.numel() / len;
  auto& plan = fft_plan<f32, c32>({len}, batch);
  plan.exec(in._data, out._data, -1);
  return true;
}

template <u32 N>
auto ifft(math::NdView<c32, N> in, math::NdView<f32, N> out) -> bool {
  if (in._size[0] / 2 + 1 != out._size[0]) {
    return false;
  }
  for (auto i = 1U; i < N; ++i) {
    if (in._size[i] != out._size[i]) {
      return false;
    }
  }

  const auto len = in._size[0];
  const auto batch = in.numel() / len;
  auto& plan = fft_plan<c32, f32>({len}, batch);
  plan.exec(in._data, out._data, 1);
  return true;
}

template auto fft(NdView<c32, 1> in, NdView<c32, 1> out) -> bool;
template auto fft(NdView<f32, 1> in, NdView<c32, 1> out) -> bool;

template auto ifft(NdView<c32, 1> in, NdView<c32, 1> out) -> bool;
template auto ifft(NdView<c32, 1> in, NdView<f32, 1> out) -> bool;
}  // namespace nct::math
