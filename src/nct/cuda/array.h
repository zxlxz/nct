#pragma once

#include "nct/math/slice.h"
#include "nct/cuda/mem.h"

namespace nct::cuda {

template <class T>
class Buffer {
  T* _ptr = nullptr;
  usize _cap = 0;
  MemType _mem = MemType::CPU;

 public:
  explicit Buffer() noexcept = default;

  ~Buffer() noexcept {
    this->reset();
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept : _ptr{other._ptr}, _cap{other._cap}, _mem{other._mem} {
    other._ptr = nullptr;
    other._cap = 0;
    other._mem = MemType::CPU;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->reset();
    _ptr = other._ptr, other._ptr = nullptr;
    _cap = other._cap, other._cap = 0;
    _mem = other._mem, other._mem = MemType::CPU;
    return *this;
  }

  static auto with_capacity(usize capacity, MemType mem_type = MemType::CPU) -> Buffer {
    auto res = Buffer{};
    res._ptr = static_cast<T*>(detail::alloc(capacity * sizeof(T), mem_type));
    res._cap = capacity;
    res._mem = mem_type;
    return res;
  }

  void reset() {
    if (_ptr != nullptr) {
      detail::dealloc(_ptr, _mem);
    }
    _ptr = nullptr;
    _cap = 0;
    _mem = MemType::CPU;
  }

  auto ptr() const noexcept -> T* {
    return _ptr;
  }

  auto cap() const noexcept -> usize {
    return _cap;
  }

  void set0(stream_t stream = nullptr) {
    detail::memset(_ptr, 0, _cap * sizeof(T), stream);
  }

  void sync_cpu() {
    if (_mem != MemType::MIXED) {
      return;
    }
    detail::msync(_ptr, _cap * sizeof(T), MemType::CPU);
  }

  void sync_gpu() {
    if (_mem != MemType::MIXED) {
      return;
    }
    return detail::msync(_ptr, _cap * sizeof(T), MemType::GPU);
  }
};

template <class T, int N = 1>
class NdArray {
  using Buf = cuda::Buffer<T>;
  using Inn = math::NdSlice<T, N>;
  using Dim = math::vec<u32, N>;

  Inn _inn{};
  Buf _buf{};

 public:
  NdArray() noexcept = default;
  ~NdArray() noexcept = default;

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  NdArray(const NdArray&) = delete;
  NdArray& operator=(const NdArray&) = delete;

  static auto with_dim(Dim dims, MemType mem_type = MemType::CPU) -> NdArray {
    const auto step = math::make_step(dims);
    const auto capacity = (&dims.x)[N - 1] * (&step.x)[N - 1];

    auto res = NdArray{};
    res._buf = Buf::with_capacity(capacity, mem_type);
    res._inn = Inn{res._buf.ptr(), dims, step};
    return res;
  }

  auto operator*() const -> Inn {
    return _inn;
  }

  auto operator[](const Dim& idx) const -> T {
    return _inn[idx];
  }

  auto operator[](const Dim& idx) -> T& {
    return _inn[idx];
  }

  auto data() const -> T* {
    return _inn._data;
  }

  auto dims() const -> const Dim& {
    return _inn._dims;
  }

  auto step() const -> const Dim& {
    return _inn._step;
  }

  auto size() const -> u32 {
    return _inn.size();
  }

  auto slice(Dim start, Dim end) const -> Inn {
    return _inn.slice(start, end);
  }

  void sync_cpu() {
    return _buf.sync_cpu();
  }

  void sync_gpu() {
    return _buf.sync_gpu();
  }

  void set0(stream_t stream = nullptr) {
    _buf.set0(stream);
  }
};

}  // namespace nct::cuda
