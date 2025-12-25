#include <cuda_runtime_api.h>
#include "nct/cuda/stream.h"

namespace nct::cuda::detail {

auto event_new() -> event_t {
  auto evt = event_t{nullptr};
  if (auto err = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming)) {
    throw Error{err};
  }
  return evt;
}

void event_del(event_t evt) {
  if (evt == nullptr) {
    return;
  }

  (void)cudaEventDestroy(evt);
}

void event_wait(event_t evt) {
  if (evt == nullptr) {
    return;
  }

  if (auto err = cudaEventSynchronize(evt)) {
    throw Error{err};
  }
}

auto stream_new() -> stream_t {
  auto strm = stream_t{nullptr};
  if (auto err = cudaStreamCreate(&strm)) {
    throw Error{err};
  }
  return strm;
}

void stream_sync(stream_t strm) {
  if (strm == nullptr) {
    return;
  }

  if (auto err = cudaStreamSynchronize(strm)) {
    throw Error{err};
  }
}

void stream_del(stream_t strm) {
  if (strm == nullptr) {
    return;
  }

  (void)cudaStreamDestroy(strm);
}

void stream_wait(stream_t strm, event_t evt) {
  if (strm == nullptr || evt == nullptr) {
    return;
  }

  if (auto err = cudaStreamWaitEvent(strm, evt, 0)) {
    throw Error{err};
  }
}

void stream_record(stream_t strm, event_t evt) {
  if (strm == nullptr || evt == nullptr) {
    return;
  }

  if (auto err = cudaEventRecord(evt, strm)) {
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

}  // namespace nct::cuda::detail
