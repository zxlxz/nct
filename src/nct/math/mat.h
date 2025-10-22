#pragma once

#include "nct/math/vec.h"

namespace nct::math {

template <class T, int N>
struct mat;

template <class T>
struct mat<T, 2> {
  using col_t = nvec<T, 2>;
  col_t c0, c1;

 public:
  __hd__ static auto from_rows(const T (&m)[2][2]) -> mat {
    return {{m[0][0], m[1][0]}, {m[0][1], m[1][1]}};
  }

  __hd__ static auto from_cols(const T (&m)[2][2]) -> mat {
    return {{m[0][0], m[0][1]}, {m[1][0], m[1][1]}};
  }

  __hd__ auto operator*(col_t v) const -> col_t {
    return {dot(c0, v), dot(c1, v)};
  }
};

template <class T>
struct mat<T, 3> {
  using col_t = nvec<T, 3>;
  col_t c0, c1, c2;

 public:
  __hd__ static auto from_rows(const T (&m)[3][3]) -> mat {
    return {
        {m[0][0], m[1][0], m[2][0]},
        {m[0][1], m[1][1], m[2][1]},
        {m[0][2], m[1][2], m[2][2]},
    };
  }

  __hd__ static auto from_cols(const T (&m)[3][3]) -> mat {
    return {
        {m[0][0], m[0][1], m[0][2]},
        {m[1][0], m[1][1], m[1][2]},
        {m[2][0], m[2][1], m[2][2]},
    };
  }

  __hd__ auto operator*(col_t v) const -> col_t {
    return {dot(c0, v), dot(c1, v), dot(c2, v)};
  }
};

template <class T>
struct mat<T, 4> {
  using col_t = nvec<T, 4>;
  col_t c0, c1, c2, c3;

 public:
  __hd__ static auto from_rows(const T (&m)[4][4]) -> mat {
    return {
        {m[0][0], m[1][0], m[2][0], m[3][0]},
        {m[0][1], m[1][1], m[2][1], m[3][1]},
        {m[0][2], m[1][2], m[2][2], m[3][2]},
        {m[0][3], m[1][3], m[2][3], m[3][3]},
    };
  }

  __hd__ static auto from_cols(const T (&m)[4][4]) -> mat {
    return {
        {m[0][0], m[0][1], m[0][2], m[0][3]},
        {m[1][0], m[1][1], m[1][2], m[1][3]},
        {m[2][0], m[2][1], m[2][2], m[2][3]},
        {m[3][0], m[3][1], m[3][2], m[3][3]},
    };
  }

  __hd__ auto operator*(col_t v) const -> col_t {
    return {dot(c0, v), dot(c1, v), dot(c2, v), dot(c3, v)};
  }

  __hd__ auto operator*(mat m) const -> mat {
    return {};
  }
};

using mat2f = mat<float, 2>;
using mat3f = mat<float, 3>;
using mat4f = mat<float, 4>;

}  // namespace nct::math
