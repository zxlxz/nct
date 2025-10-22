#pragma once

#include "nct/math/mod.h"

namespace nct::math {

template <class T, u32 N>
struct nvec;

template <class T>
struct alignas(sizeof(T) * 1) nvec<T, 1> {
  using item_t = T;

  T x;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> nvec<U, 1> {
    return {static_cast<U>(x)};
  }

  __hd__ auto dot(nvec v) const noexcept -> T {
    return x * v.x;
  }

  __hd__ auto operator,(T y) const noexcept -> nvec<T, 2> {
    return {x, y};
  }

  __hd__ auto operator+(nvec v) const noexcept -> nvec {
    return {x + v.x};
  }

  __hd__ auto operator-(nvec v) const noexcept -> nvec {
    return {x - v.x};
  }

  __hd__ auto operator*(nvec v) const noexcept -> nvec {
    return {x * v.x};
  }

  __hd__ auto operator<(nvec v) const noexcept -> bool {
    return x < v.x;
  }

  __hd__ friend auto operator*(T scale, nvec v) noexcept -> nvec {
    return {scale * v.x};
  }
};

template <class T>
struct alignas(sizeof(T) * 2) nvec<T, 2> {
  using item_t = T;

  T x, y;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> nvec<U, 2> {
    return {static_cast<U>(x), static_cast<U>(y)};
  }

  __hd__ auto operator,(T z) const noexcept -> nvec<T, 3> {
    return {x, y, z};
  }

  __hd__ auto operator+(nvec v) const noexcept -> nvec {
    return {x + v.x, y + v.y};
  }

  __hd__ auto operator-(nvec v) const noexcept -> nvec {
    return {x - v.x, y - v.y};
  }

  __hd__ auto operator*(nvec v) const noexcept -> nvec {
    return {x * v.x, y * v.y};
  }

  __hd__ auto operator/(nvec v) const noexcept -> nvec {
    return {x / v.x, y / v.y};
  }

  __hd__ auto operator<(nvec v) const noexcept -> bool {
    return x < v.x && y < v.y;
  }

  __hd__ friend auto operator*(T scale, nvec v) noexcept -> nvec {
    return {scale * v.x, scale * v.y};
  }

  __hd__ friend auto operator/(T scale, nvec v) noexcept -> nvec {
    return {scale / v.x, scale / v.y};
  }
};

template <class T>
struct alignas(sizeof(T) * 1) nvec<T, 3> {
  using item_t = T;
  T x, y, z;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> nvec<U, 3> {
    return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
  }

  __hd__ auto operator,(T w) const noexcept -> nvec<T, 4> {
    return {x, y, z, w};
  }

  __hd__ auto operator+(nvec v) const noexcept -> nvec {
    return {x + v.x, y + v.y, z + v.z};
  }

  __hd__ auto operator-(nvec v) const noexcept -> nvec {
    return {x - v.x, y - v.y, z - v.z};
  }

  __hd__ auto operator*(nvec v) const noexcept -> nvec {
    return {x * v.x, y * v.y, z * v.z};
  }

  __hd__ auto operator/(nvec v) const noexcept -> nvec {
    return {x / v.x, y / v.y, z / v.z};
  }

  __hd__ auto operator<(nvec v) const noexcept -> bool {
    return x < v.x && y < v.y && z < v.z;
  }

  __hd__ friend auto operator*(T scale, nvec v) noexcept -> nvec {
    return {scale * v.x, scale * v.y, scale * v.z};
  }

  void fmt(auto& f) const {
    return f.write_fmt("[{}, {}, {}]", x, y, z);
  }
};

template <class T>
struct alignas(sizeof(T) * 4) nvec<T, 4> {
  using item_t = T;
  T x, y, z, w;

 public:
  __hd__ auto as_vec3() const noexcept -> nvec<T, 3> {
    return {x, y, z};
  }

  __hd__ auto dot(nvec v) const noexcept -> T {
    return x * v.x + y * v.y + z * v.z + w * v.w;
  }

  __hd__ auto operator+(nvec v) const noexcept -> nvec {
    return {x + v.x, y + v.y, z + v.z, w + v.w};
  }

  __hd__ auto operator-(nvec v) const noexcept -> nvec {
    return {x - v.x, y - v.y, z - v.z, w - v.w};
  }

  __hd__ auto operator*(nvec v) const noexcept -> nvec {
    return {x * v.x, y * v.y, z * v.z, w * v.w};
  }

  __hd__ auto operator<(nvec v) const noexcept -> bool {
    return x < v.x && y < v.y && z < v.z && w < v.w;
  }

  __hd__ friend auto operator*(T scale, nvec v) noexcept -> nvec {
    return {scale * v.x, scale * v.y, scale * v.z, scale * v.w};
  }
};

using f32x1 = nvec<float, 1>;
using f32x2 = nvec<float, 2>;
using f32x3 = nvec<float, 3>;
using f32x4 = nvec<float, 4>;

using i32x1 = nvec<int, 1>;
using i32x2 = nvec<int, 2>;
using i32x3 = nvec<int, 3>;
using i32x4 = nvec<int, 4>;

using u32x1 = nvec<unsigned int, 1>;
using u32x2 = nvec<unsigned int, 2>;
using u32x3 = nvec<unsigned int, 3>;
using u32x4 = nvec<unsigned int, 4>;

__hd__ inline auto len(const nvec<float, 2>& v) noexcept -> float {
  return sqrtf(v.x * v.x + v.y * v.y);
}

__hd__ inline auto len(const nvec<float, 3>& v) noexcept -> float {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__hd__ inline auto rot(f32 a) noexcept -> f32x2 {
#ifdef __CUDACC__
  auto c = 0.f;
  auto s = 0.f;
  ::sincosf(a, &s, &c);
  return {c, s};
#else
  return {cosf(a), sinf(a)};
#endif
}

__hd__ inline auto norm(f32x2 v) noexcept -> f32x2 {
  const auto s = sqrtf(v.x * v.x + v.y * v.y);
  return {v.x / s, v.y / s};
}

__hd__ inline auto norm(f32x3 v) noexcept -> f32x3 {
  const auto s = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  return {v.x / s, v.y / s, v.z / s};
}

__hd__ inline auto dot(f32x2 a, f32x2 b) noexcept -> f32 {
  return a.x * b.x + a.y * b.y;
}

__hd__ inline auto dot(f32x3 a, f32x3 b) noexcept -> f32 {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__hd__ inline auto cross(f32x2 a, f32x2 b) noexcept -> f32 {
  return a.x * b.y - a.y * b.x;
}

__hd__ inline auto cross(f32x3 a, f32x3 b) noexcept -> f32x3 {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}  // namespace nct::math
