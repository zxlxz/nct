#pragma once

#include "nct/core.h"

struct cudaArray;
struct CUstream_st;

namespace nct::cuda {

using arr_t = struct ::cudaArray*;
using stream_t = CUstream_st*;

enum class MemType {
  CPU = 0,
  GPU = 1,
  MIXED = 2,
};

struct Ptr3D {
  void* ptr = nullptr;
  u32 size = 1;
  u32 dims[3] = {0, 0, 0};
  u32 step[3] = {0, 0, 0};
};

auto alloc(MemType type, usize size) -> void*;

auto dealloc(MemType type, void* ptr) -> void;

auto prefetch(MemType type, void* ptr, usize size) -> void;

void fill_bytes(void* ptr, u8 val, usize size, stream_t stream);

void copy_bytes(void* dst, const void* src, usize size, stream_t stream);

void copy_3d(Ptr3D src, Ptr3D dst, stream_t stream = nullptr);

template <class T>
void zero(T* buff, usize size, stream_t stream = nullptr) {
  cuda::fill_bytes(buff, 0, size * sizeof(T), stream);
}

template <class T>
void copy(const T* src, T* dst, usize size, stream_t stream = nullptr) {
  cuda::copy_bytes(dst, src, size * sizeof(T), stream);
}

template <class T>
class RawBuf {
  T* _ptr = nullptr;
  usize _cap = 0;

 public:
  explicit RawBuf() noexcept = default;

  ~RawBuf() noexcept {
    if (_ptr != nullptr) {
      cuda::dealloc(MemType::GPU, _ptr);
    }
  }

  RawBuf(const RawBuf&) = delete;
  RawBuf& operator=(const RawBuf&) = delete;

  RawBuf(RawBuf&& other) noexcept : _ptr{other._ptr}, _cap{other._cap} {
    other._ptr = nullptr;
    other._cap = 0;
  }

  RawBuf& operator=(RawBuf&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (_ptr != nullptr) {
      cuda::dealloc(MemType::GPU, _ptr);
    }
    _ptr = mem::take(other._ptr);
    _cap = mem::take(other._cap);
    return *this;
  }

  static auto with_capacity(usize capacity, MemType mtype = {}) -> RawBuf<T> {
    auto res = RawBuf<T>{};
    res._ptr = static_cast<T*>(cuda::alloc(mtype, capacity * sizeof(T)));
    res._cap = capacity;
    return res;
  }

  auto ptr() const noexcept -> T* {
    return _ptr;
  }

  auto cap() const noexcept -> usize {
    return _cap;
  }

  void sync_cpu() {
    cuda::prefetch(MemType::CPU, _ptr, _cap * sizeof(T));
  }

  void sync_gpu() {
    cuda::prefetch(MemType::GPU, _ptr, _cap * sizeof(T));
  }

#ifndef __CUDACC__
  auto as_bytes() const -> sfc::Slice<const u8> {
    return {reinterpret_cast<const u8*>(_ptr), _cap * sizeof(T)};
  }

  auto as_bytes_mut() -> sfc::Slice<u8> {
    return {reinterpret_cast<u8*>(_ptr), _cap * sizeof(T)};
  }
#endif
};

}  // namespace nct::cuda

namespace nct {
using cuda::MemType;
using cuda::RawBuf;
}  // namespace nct
