#pragma once

#include "nct/math/mod.h"

namespace nct::math {

template <class T, int N>
struct vec;

template <class T>
struct alignas(sizeof(T) * 1) vec<T, 1> {
  using item_t = T;

  T x;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> vec<U, 1> {
    return {static_cast<U>(x)};
  }

  __hd__ auto dot(vec v) const noexcept -> T {
    return x * v.x;
  }

  __hd__ auto operator,(T y) const noexcept -> vec<T, 2> {
    return {x, y};
  }

  __hd__ auto operator+(vec v) const noexcept -> vec {
    return {x + v.x};
  }

  __hd__ auto operator-(vec v) const noexcept -> vec {
    return {x - v.x};
  }

  __hd__ auto operator*(vec v) const noexcept -> vec {
    return {x * v.x};
  }

  __hd__ auto operator<(vec v) const noexcept -> bool {
    return x < v.x;
  }

  __hd__ friend auto operator*(T scale, vec v) noexcept -> vec {
    return {scale * v.x};
  }
};

template <class T>
struct alignas(sizeof(T) * 2) vec<T, 2> {
  using item_t = T;

  T x, y;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> vec<U, 2> {
    return {static_cast<U>(x), static_cast<U>(y)};
  }

  __hd__ auto operator,(T z) const noexcept -> vec<T, 3> {
    return {x, y, z};
  }

  __hd__ auto operator+(vec v) const noexcept -> vec {
    return {x + v.x, y + v.y};
  }

  __hd__ auto operator-(vec v) const noexcept -> vec {
    return {x - v.x, y - v.y};
  }

  __hd__ auto operator*(vec v) const noexcept -> vec {
    return {x * v.x, y * v.y};
  }

  __hd__ auto operator/(vec v) const noexcept -> vec {
    return {x / v.x, y / v.y};
  }

  __hd__ auto operator<(vec v) const noexcept -> bool {
    return x < v.x && y < v.y;
  }

  __hd__ friend auto operator*(T scale, vec v) noexcept -> vec {
    return {scale * v.x, scale * v.y};
  }

  __hd__ friend auto operator/(T scale, vec v) noexcept -> vec {
    return {scale / v.x, scale / v.y};
  }
};

template <class T>
struct alignas(sizeof(T) * 1) vec<T, 3> {
  using item_t = T;
  T x, y, z;

 public:
  template <class U>
  __hd__ auto as() const noexcept -> vec<U, 3> {
    return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
  }

  __hd__ auto operator,(T w) const noexcept -> vec<T, 4> {
    return {x, y, z, w};
  }

  __hd__ auto operator+(vec v) const noexcept -> vec {
    return {x + v.x, y + v.y, z + v.z};
  }

  __hd__ auto operator-(vec v) const noexcept -> vec {
    return {x - v.x, y - v.y, z - v.z};
  }

  __hd__ auto operator*(vec v) const noexcept -> vec {
    return {x * v.x, y * v.y, z * v.z};
  }

  __hd__ auto operator/(vec v) const noexcept -> vec {
    return {x / v.x, y / v.y, z / v.z};
  }

  __hd__ auto operator<(vec v) const noexcept -> bool {
    return x < v.x && y < v.y && z < v.z;
  }

  __hd__ friend auto operator*(T scale, vec v) noexcept -> vec {
    return {scale * v.x, scale * v.y, scale * v.z};
  }

  void fmt(auto& f) const {
    return f.write_fmt("[{}, {}, {}]", x, y, z);
  }
};

template <class T>
struct alignas(sizeof(T) * 4) vec<T, 4> {
  using item_t = T;
  T x, y, z, w;

 public:
  __hd__ auto as_vec3() const noexcept -> vec<T, 3> {
    return {x, y, z};
  }

  __hd__ auto dot(vec v) const noexcept -> T {
    return x * v.x + y * v.y + z * v.z + w * v.w;
  }

  __hd__ auto operator+(vec v) const noexcept -> vec {
    return {x + v.x, y + v.y, z + v.z, w + v.w};
  }

  __hd__ auto operator-(vec v) const noexcept -> vec {
    return {x - v.x, y - v.y, z - v.z, w - v.w};
  }

  __hd__ auto operator*(vec v) const noexcept -> vec {
    return {x * v.x, y * v.y, z * v.z, w * v.w};
  }

  __hd__ auto operator<(vec v) const noexcept -> bool {
    return x < v.x && y < v.y && z < v.z && w < v.w;
  }

  __hd__ friend auto operator*(T scale, vec v) noexcept -> vec {
    return {scale * v.x, scale * v.y, scale * v.z, scale * v.w};
  }
};

using vec2i = vec<int, 2>;
using vec3i = vec<int, 3>;
using vec4i = vec<int, 4>;

using vec2f = vec<float, 2>;
using vec3f = vec<float, 3>;
using vec4f = vec<float, 4>;

using f32x2 = vec<float, 2>;
using f32x3 = vec<float, 3>;
using f32x4 = vec<float, 4>;

using i32x2 = vec<int, 2>;
using i32x3 = vec<int, 3>;
using i32x4 = vec<int, 4>;

using u32x2 = vec<unsigned int, 2>;
using u32x3 = vec<unsigned int, 3>;
using u32x4 = vec<unsigned int, 4>;

__hd__ inline auto len(const vec<float, 2>& v) noexcept -> float {
  return sqrtf(v.x * v.x + v.y * v.y);
}

__hd__ inline auto len(const vec<float, 3>& v) noexcept -> float {
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
  const auto k = rsqrtf(v.x * v.x + v.y * v.y);
  return k * v;
}

__hd__ inline auto norm(f32x3 v) noexcept -> f32x3 {
  const auto k = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  return k * v;
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
