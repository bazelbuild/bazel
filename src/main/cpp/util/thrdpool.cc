// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/main/cpp/util/thrdpool.h"

namespace blaze_util {
namespace threads {

// Tries removing the item from the front of `queue`, returns true upon success.
// The queue is guarded by `queue_mutex`. If the function returns true, the
// popped iteam is stored in `result`.
static bool Pop(std::queue<std::function<void()> >* queue,
                std::mutex* queue_mutex, std::function<void()>* result) {
  std::lock_guard<std::mutex> guard(*queue_mutex);
  if (queue->empty()) {
    return false;
  }
  *result = queue->front();
  queue->pop();
  return true;
}

static void ThreadFunc(std::queue<std::function<void()> >* queue,
                       std::mutex* queue_mutex, std::atomic_bool* pushing) {
  std::function<void()> func;
  bool running = true;
  while (true) {
    if (!*pushing) {
      // Do not return yet: the pushing thread may just have finished pushing
      // the queue but there may be still unprocessed work items. Let the drain
      // cycle run one last time.
      running = false;
    }
    while (Pop(queue, queue_mutex, &func)) {
      // Drain as many work items as possible.
      func();
    }
    if (running) {
      // Temporarily drained the queue. Let the pushing thread push more work
      // items.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } else {
      // Fully drained the queue, return.
      break;
    }
  }
}

ThreadPool::ThreadPool(const size_t size)
    : size_(size), pushing_(true), threads_(new std::thread[size]) {
  for (size_t i = 0; i < size_; i++) {
    threads_[i] = std::thread(ThreadFunc, &queue_, &queue_mutex_, &pushing_);
  }
}

ThreadPool::~ThreadPool() { Join(); }

bool ThreadPool::Push(std::function<void()> func) {
  if (!pushing_) {
    return false;
  }
  std::lock_guard<std::mutex> guard(queue_mutex_);
  queue_.push(func);
  return true;
}

void ThreadPool::Join() {
  if (pushing_) {
    pushing_ = false;
    for (size_t i = 0; i < size_; i++) {
      threads_[i].join();
    }
  }
}

}  // namespace threads
}  // namespace blaze_util
