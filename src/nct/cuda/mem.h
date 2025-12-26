#pragma once

#include "nct/core.h"

namespace nct::cuda {

auto err_name(int code) -> const char*;

auto alloc_cpu(size_t size) -> void*;
void dealloc_cpu(void* ptr);

auto alloc_gpu(size_t size) -> void*;
void dealloc_gpu(void* ptr);

auto alloc_uma(size_t size) -> void*;
void dealloc_uma(void* ptr);

void prefetch_cpu(void* ptr, size_t size);
void prefetch_gpu(void* ptr, size_t size);

void write_bytes(void* ptr, u8 val, size_t size);
void copy_bytes(const void* src, void* dst, size_t size);

enum class MemType {
  CPU = 0,
  GPU = 1,
  UMA = 2,
};

struct Error {
  int _code = 0;

 public:
  const char* what() const noexcept {
    return cuda::err_name(_code);
  }
};

struct Alloc {
  using enum MemType;
  MemType _type = MemType::CPU;

 public:
  Alloc(MemType t = MemType::CPU) : _type(t) {}

  auto alloc(size_t size) -> void* {
    switch (_type) {
      case MemType::CPU: return cuda::alloc_cpu(size);
      case MemType::GPU: return cuda::alloc_gpu(size);
      case MemType::UMA: return cuda::alloc_uma(size);
      default:           return nullptr;
    }
  }

  void dealloc(void* ptr) {
    switch (_type) {
      case MemType::CPU: cuda::dealloc_cpu(ptr); break;
      case MemType::GPU: cuda::dealloc_gpu(ptr); break;
      case MemType::UMA: cuda::dealloc_uma(ptr); break;
      default:           break;
    }
  }

  void sync_cpu(void* ptr, size_t size) {
    if (_type == MemType::UMA) {
      cuda::prefetch_cpu(ptr, size);
    }
  }

  void sync_gpu(void* ptr, size_t size) {
    if (_type == MemType::UMA) {
      cuda::prefetch_gpu(ptr, size);
    }
  }
};

}  // namespace nct::cuda
