#pragma once

#include "nct/core.h"

struct CUstream_st;
struct CUevent_st;

namespace nct::cuda {

using event_t = CUevent_st*;
using stream_t = CUstream_st*;

namespace detail {
auto err_name(int) -> const char*;

auto event_new() -> event_t;
void event_del(event_t);
void event_wait(event_t);

auto stream_new() -> stream_t;
void stream_del(stream_t);
void stream_sync(stream_t);
void stream_wait(stream_t, event_t);
void stream_record(stream_t, event_t);

auto stream_current() -> stream_t;
void stream_push(stream_t);
void stream_pop();

}  // namespace detail

class Event {
  friend class Stream;
  event_t _raw = nullptr;

 public:
  explicit Event() : _raw{detail::event_new()} {}

  ~Event() noexcept {
    if (_raw != nullptr) {
      detail::event_del(_raw);
    }
  }

  Event(Event&& other) noexcept : _raw{mem::take(other._raw)} {}

  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      mem::swap(_raw, other._raw);
    }
    return *this;
  }

  auto raw() const -> event_t {
    return _raw;
  }

  void wait() {
    detail::event_wait(_raw);
  }
};

class Stream {
  stream_t _raw = nullptr;

 public:
  explicit Stream() : _raw{detail::stream_new()} {}

  ~Stream() noexcept {
    if (_raw != nullptr) {
      detail::stream_del(_raw);
    }
  }

  Stream(Stream&& other) noexcept : _raw{mem::take(other._raw)} {}

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      mem::swap(_raw, other._raw);
    }
    return *this;
  }

  auto raw() const -> stream_t {
    return _raw;
  }

  void sync() {
    detail::stream_sync(_raw);
  }

  void wait(const Event& evt) {
    detail::stream_wait(_raw, evt.raw());
  }

  auto record() -> Event {
    auto evt = Event{};
    detail::stream_record(_raw, evt.raw());
    return evt;
  }

 public:
  void run(auto& f) {
    detail::stream_push(_raw);
    f();
    detail::stream_pop();
  }
};

}  // namespace nct::cuda
