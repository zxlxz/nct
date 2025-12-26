#include <cuda_runtime_api.h>
#include "nct/cuda/stream.h"
#include "nct/cuda/mem.h"

namespace nct::cuda {

auto event_new() -> event_t {
  auto event = event_t{nullptr};
  if (auto err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming)) {
    throw Error{err};
  }
  return event;
}

void event_del(event_t event) {
  if (event == nullptr) {
    return;
  }

  (void)cudaEventDestroy(event);
}

void event_wait(event_t event) {
  if (event == nullptr) {
    return;
  }

  if (auto err = cudaEventSynchronize(event)) {
    throw Error{err};
  }
}

auto stream_new() -> stream_t {
  auto stream = stream_t{nullptr};
  if (auto err = cudaStreamCreate(&stream)) {
    throw Error{err};
  }
  return stream;
}

void stream_sync(stream_t stream) {
  if (stream == nullptr) {
    return;
  }

  if (auto err = cudaStreamSynchronize(stream)) {
    throw Error{err};
  }
}

void stream_del(stream_t stream) {
  if (stream == nullptr) {
    return;
  }

  (void)cudaStreamDestroy(stream);
}

void stream_wait(stream_t stream, event_t event) {
  if (stream == nullptr || event == nullptr) {
    return;
  }

  if (auto err = cudaStreamWaitEvent(stream, event, 0)) {
    throw Error{err};
  }
}

void stream_record(stream_t stream, event_t event) {
  if (stream == nullptr || event == nullptr) {
    return;
  }

  if (auto err = cudaEventRecord(event, stream)) {
    throw Error{err};
  }
}

auto stream_stack() -> Vec<stream_t>& {
  static thread_local auto stack = Vec<stream_t>{};
  return stack;
}

auto stream_current() -> stream_t {
  auto& stack = stream_stack();
  if (stack.is_empty()) {
    return nullptr;
  }
  return stack[stack.len() - 1];
}

void stream_push(stream_t stream) {
  auto& stack = stream_stack();
  stack.push(stream);
}

void stream_pop() {
  auto& stack = stream_stack();
  (void)stack.pop();
}

}  // namespace nct::cuda
