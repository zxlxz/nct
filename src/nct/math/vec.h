#pragma once

#include "nct/math/mod.h"

namespace nct::math {

template <class T, u32 N>
struct vec;

template <class T>
struct alignas(sizeof(T) * 2) vec<T, 2> {
  using arr_t = T[2];
  T x, y;

  operator const arr_t&() const {
    return reinterpret_cast<const arr_t&>(*this);
  }
};

template <class T>
struct alignas(sizeof(T) * 1) vec<T, 3> {
  using arr_t = T[3];
  T x, y, z;

  operator const arr_t&() const {
    return reinterpret_cast<const arr_t&>(*this);
  }
};

template <class T>
struct alignas(sizeof(T) * 4) vec<T, 4> {
  using arr_t = T[4];
  T x, y, z, w;

  operator const arr_t&() const {
    return reinterpret_cast<const arr_t&>(*this);
  }
};

using vec2i = vec<i32, 2>;
using vec2u = vec<u32, 2>;
using vec2f = vec<f32, 2>;

using vec3i = vec<i32, 3>;
using vec3u = vec<u32, 3>;
using vec3f = vec<f32, 3>;

template <class T, u32 N>
__hd__ inline auto operator-(const vec<T, N>& v) noexcept -> vec<T, N> {
  static_assert(N >= 1 && N <= 3, "-v: N out of range");
  if constexpr (N == 1) {
    return {-v.x};
  } else if constexpr (N == 2) {
    return {-v.x, -v.y};
  } else if constexpr (N == 3) {
    return {-v.x, -v.y, -v.z};
  }
}

template <class T, u32 N>
__hd__ inline auto operator*(f32 k, const vec<T, N>& v) noexcept -> vec<f32, N> {
  static_assert(N >= 1 && N <= 3, "k*v: N out of range");
  if constexpr (N == 1) {
    return {k * v.x};
  } else if constexpr (N == 2) {
    return {k * v.x, k * v.y};
  } else if constexpr (N == 3) {
    return {k * v.x, k * v.y, k * v.z};
  }
}

template <class T, u32 N>
__hd__ inline auto operator+(const vec<T, N>& a, const vec<T, N>& b) noexcept -> vec<T, N> {
  static_assert(N >= 1 && N <= 3, "a+b: N out of range");
  if constexpr (N == 1) {
    return {a.x + b.x};
  } else if constexpr (N == 2) {
    return {a.x + b.x, a.y + b.y};
  } else if constexpr (N == 3) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }
}

template <class T, u32 N>
__hd__ inline auto operator-(const vec<T, N>& a, const vec<T, N>& b) noexcept -> vec<T, N> {
  static_assert(N >= 1 && N <= 3, "a-b: N out of range");
  if constexpr (N == 1) {
    return {a.x - b.x};
  } else if constexpr (N == 2) {
    return {a.x - b.x, a.y - b.y};
  } else if constexpr (N == 3) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }
}

template <class T, u32 N>
__hd__ inline auto operator*(const vec<T, N>& a, const vec<T, N>& b) noexcept -> vec<T, N> {
  static_assert(N >= 1 && N <= 3, "a*b: N out of range");
  if constexpr (N == 1) {
    return {a.x * b.x};
  } else if constexpr (N == 2) {
    return {a.x * b.x, a.y * b.y};
  } else if constexpr (N == 3) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
  }
}

template <class T, u32 N>
__hd__ inline auto operator/(const vec<T, N>& a, const vec<T, N>& b) noexcept -> vec<T, N> {
  static_assert(N >= 1 && N <= 3, "a/b: N out of range");
  if constexpr (N == 1) {
    return {a.x / b.x};
  } else if constexpr (N == 2) {
    return {a.x / b.x, a.y / b.y};
  } else if constexpr (N == 3) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
  }
}

template <u32 N>
__hd__ inline auto operator*(const vec<u32, N>& a, const vec<f32, N>& b) noexcept -> vec<f32, N> {
  static_assert(N >= 1 && N <= 3, "a*b: N out of range");
  if constexpr (N == 1) {
    return {static_cast<f32>(a.x) * b.x};
  } else if constexpr (N == 2) {
    return {static_cast<f32>(a.x) * b.x, static_cast<f32>(a.y) * b.y};
  } else if constexpr (N == 3) {
    return {static_cast<f32>(a.x) * b.x, static_cast<f32>(a.y) * b.y, static_cast<f32>(a.z) * b.z};
  }
}

template <u32 N>
__hd__ inline auto len(const vec<f32, N>& v) noexcept -> float {
  static_assert(N >= 1 && N <= 3, "dot: N out of range");
  if constexpr (N == 1) {
    return fabsf(v.x);
  } else if constexpr (N == 2) {
    return sqrtf(v.x * v.x + v.y * v.y);
  } else if constexpr (N == 3) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  }
}

template <u32 N>
__hd__ inline auto norm(const vec<f32, N>& v) noexcept -> vec<f32, N> {
  static_assert(N >= 1 && N <= 3, "dot: N out of range");
  const auto s = len(v);
  return (1.0f / s) * v;
}

template <u32 N>
__hd__ inline auto dot(const vec<f32, N>& a, const vec<f32, N>& b) noexcept -> float {
  static_assert(N >= 1 && N <= 3, "dot: N out of range");
  if constexpr (N == 1) {
    return a.x * b.x;
  } else if constexpr (N == 2) {
    return a.x * b.x + a.y * b.y;
  } else if constexpr (N == 3) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
}

template <u32 N>
__hd__ inline auto cross(const vec<f32, N>& a, const vec<f32, N>& b) noexcept -> vec<f32, N> {
  static_assert(N >= 1 && N <= 3, "cross: N out of range");
  if constexpr (N == 1) {
    return {0.0f};
  } else if constexpr (N == 2) {
    return a.x * b.y - a.y * b.x;
  } else if constexpr (N == 3) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
  }
}

__hd__ inline auto rot(f32 a) noexcept -> vec2f {
#ifdef __CUDACC__
  auto c = 0.f;
  auto s = 0.f;
  ::sincosf(a, &s, &c);
  return {c, s};
#else
  return {cosf(a), sinf(a)};
#endif
}

}  // namespace nct::math

namespace nct {
using math::vec2i;
using math::vec2u;
using math::vec2f;

using math::vec3i;
using math::vec3u;
using math::vec3f;
}  // namespace nct
