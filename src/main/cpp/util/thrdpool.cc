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

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {
namespace threads {

// Main function of worker threads: it pops items from the `queue` as long as
// `pushing` is true. Once `pushing` is false, the function drains the remaining
// elements from `queue` then returns.
static void ThreadFunc(std::queue<std::function<void()> >* queue,
                       std::mutex* queue_mutex,
                       std::condition_variable* queue_filled,
                       std::atomic_bool* pushing) {
  bool terminate = false;
  while (true) {
    std::unique_lock<std::mutex> lock(*queue_mutex);
    // Drain cycle: pop tasks from the queue as long as the queue is filled.
    while (!queue->empty()) {
      std::function<void()> task = queue->front();
      queue->pop();
      lock.unlock();  // let other threads pop tasks or Push() or Join()
      task();
      lock.lock();  // lock again to check `queue` or to start waiting
    }
    if (terminate) {
      // Drained the queue and `pushing` was false before the last drain cycle
      // so no more tasks could have arrived.
      return;
    } else if (*pushing) {
      // Temporarily drained the queue. Wait for more tasks or for Join().
      queue_filled->wait(lock);
    } else {
      // No more tasks can be pushed. Don't return yet, let the drain cycle run
      // once more, in case Push() raced with Join().
      terminate = true;
    }
  }
}

ThreadPool::ThreadPool(const size_t size)
    : size_(size), pushing_(true), threads_(new std::thread[size]) {
  for (size_t i = 0; i < size_; i++) {
    threads_[i] = std::thread(ThreadFunc, &queue_, &queue_mutex_,
                              &queue_filled_, &pushing_);
  }
}

ThreadPool::~ThreadPool() { Join(); }

bool ThreadPool::Push(const std::function<void()>& task) {
  // Acquire the lock to avoid racing with Join() and to mutate the queue.
  std::unique_lock<std::mutex> lock(queue_mutex_);
  if (!pushing_) {
    return false;  // unique_lock destructor unlocks the lock
  }
  queue_.push(task);
  lock.unlock();
  // Benign race condition: another thread may call Join() now, but the task is
  // already on the queue and worker threads drain the queue once more after
  // `pushing_` became false.
  queue_filled_.notify_one();
  return true;
}

bool ThreadPool::Join() {
  // Acquire the lock to avoid racing with Push().
  std::unique_lock<std::mutex> lock(queue_mutex_);
  if (!pushing_.exchange(false)) {
    return false;  // unique_lock destructor unlocks the lock
  }
  lock.unlock();  // let worker threads pop tasks from the queue
  // Benign race condition: another thread may call Push() now, but `pushing_`
  // is false so Push() will fail.
  queue_filled_.notify_all();
  for (size_t i = 0; i < size_; i++) {
    threads_[i].join();
  }
  if (!queue_.empty()) {
    // This is a bug: the worker threads should have drained the queue.
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "ThreadPool error: " << queue_.size()
        << " task(s) remained in queue after Join()";
  }
  return true;
}

}  // namespace threads
}  // namespace blaze_util
