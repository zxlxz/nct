#include <cuda_runtime_api.h>
#include "nct/cuda/mod.h"

namespace nct::cuda {

const char* Error::what() const noexcept {
  const auto name = cudaGetErrorName(static_cast<cudaError_t>(_code));
  return name;
}

auto Device::count() -> int {
  auto cnt = 0;
  if (auto err = cudaGetDeviceCount(&cnt)) {
    throw Error{err};
  }
  return cnt;
}

auto Device::current() -> int {
  auto id = 0;
  if (auto err = cudaGetDevice(&id)) {
    throw Error{err};
  }
  return id;
}

void Device::set(int id) {
  if (auto err = cudaSetDevice(id)) {
    throw Error{err};
  }
}

auto Device::info(int id) -> Info {
  auto prop = cudaDeviceProp{};
  if (auto err = cudaGetDeviceProperties(&prop, id)) {
    throw Error{err};
  }

  const auto res = Info{
      .memory_size = prop.totalGlobalMem,
      .sm_count = static_cast<u32>(prop.multiProcessorCount),

  };
  return res;
}

Event::Event() {
  static const auto flags = cudaEventDisableTiming;
  if (auto err = cudaEventCreateWithFlags(&_raw, flags)) {
    throw Error{err};
  }
}

Event::~Event() noexcept {
  if (_raw == nullptr) {
    return;
  }

  (void)cudaEventDestroy(_raw);
}

Event::Event(Event&& other) noexcept : _raw{other._raw} {
  other._raw = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  auto tmp = Event{static_cast<Event&&>(*this)};
  _raw = other._raw;
  other._raw = nullptr;

  return *this;
}

auto Event::id() const -> event_t {
  return _raw;
}

void Event::wait() {
  if (_raw == nullptr) {
    return;
  }

  if (auto err = cudaEventSynchronize(_raw)) {
    throw Error{err};
  }
}

Stream::Stream() {
  if (auto err = cudaStreamCreate(&_raw)) {
    throw Error{err};
  }
}

Stream::~Stream() noexcept {
  if (_raw == nullptr) {
    return;
  }

  (void)cudaStreamDestroy(_raw);
}

Stream::Stream(Stream&& other) noexcept : _raw{other._raw} {
  other._raw = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  auto tmp = Stream{static_cast<Stream&&>(*this)};
  _raw = other._raw;
  other._raw = nullptr;

  return *this;
}

auto Stream::id() const -> stream_t {
  return _raw;
}

void Stream::sync() {
  if (_raw == nullptr) {
    return;
  }

  if (auto err = ::cudaStreamSynchronize(_raw)) {
    throw Error{err};
  }
}

auto Stream::record() -> Event {
  auto event = Event{};

  if (auto err = cudaEventRecord(event._raw, _raw)) {
    throw Error{err};
  }
  return event;
}

void Stream::wait(const Event& event) {
  if (_raw == nullptr || event._raw == nullptr) {
    return;
  }

  if (auto err = cudaStreamWaitEvent(_raw, event._raw, 0)) {
    throw Error{err};
  }
}

}  // namespace nct::cuda
