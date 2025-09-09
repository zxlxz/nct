#pragma once

#include "nct/core/mod.h"

namespace nct {

enum class MemType {
  CPU = 0,
  GPU = 1,
  MIXED = 2,
};

struct Alloc {
  MemType _type = MemType::CPU;

 public:
  Alloc(MemType type = MemType::CPU) : _type{type} {}

  auto alloc(usize size) -> void*;
  void dealloc(void* ptr);
  void sync_cpu(void* ptr, usize size);
  void sync_gpu(void* ptr, usize size);
};

template <class T, class A = Alloc>
class RawBuf {
  T* _ptr = nullptr;
  usize _cap = 0;
  A _a = {};

 public:
  explicit RawBuf() noexcept = default;

  ~RawBuf() noexcept {
    this->reset();
  }

  RawBuf(const RawBuf&) = delete;
  RawBuf& operator=(const RawBuf&) = delete;

  RawBuf(RawBuf&& other) noexcept : _ptr{other._ptr}, _cap{other._cap}, _a{mem::move(other._a)} {
    other._ptr = nullptr;
    other._cap = 0;
  }

  RawBuf& operator=(RawBuf&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->reset();
    _ptr = mem::take(other._ptr);
    _cap = other._cap;
    _a = mem::move(other._a);
    return *this;
  }

  static auto with_capacity(usize capacity, Alloc alloc = {}) -> RawBuf<T, A> {
    auto res = RawBuf<T, A>{};
    res._ptr = static_cast<T*>(alloc.alloc(capacity * sizeof(T)));
    res._cap = capacity;
    res._a = mem::move(alloc);
    return res;
  }

  void reset() {
    if (_ptr != nullptr) {
      _a.dealloc(_ptr);
    }
    _ptr = nullptr;
    _cap = 0;
  }

  auto ptr() const noexcept -> T* {
    return _ptr;
  }

  auto cap() const noexcept -> usize {
    return _cap;
  }

  void sync_cpu() {
    return _a.sync_cpu(_ptr, _cap * sizeof(T));
  }

  void sync_gpu() {
    return _a.sync_gpu(_ptr, _cap * sizeof(T));
  }
};

}  // namespace nct
