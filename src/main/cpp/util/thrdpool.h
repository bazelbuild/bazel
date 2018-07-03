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
#ifndef BAZEL_SRC_MAIN_CPP_UTIL_THRDPOOL_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_THRDPOOL_H_

#include <stddef.h>  // size_t

#include <atomic>
#include <condition_variable>  // NOLINT
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT

namespace blaze_util {
namespace threads {

// Fixed-size thread pool.
// Completes each Push()'d task on an async thread.
class ThreadPool {
 public:
  // Creates this ThreadPool, allocating `size` many threads.
  ThreadPool(const size_t size);

  // Destructor. Also Join()'s the pool.
  ~ThreadPool();

  // Pushes a task to the work queue.
  // One of the async threads will complete the task.
  // Returns true if the task was pushed to the queue.
  // Has no effect and returns false if the pool is Join()'ing or already has.
  bool Push(const std::function<void()>& task);

  // Blocks until worker threads complete all tasks in the queue and terminate.
  // Returns true once all threads are joined. No more tasks may be Push()'d to
  // the queue afterwards. Subsequent calls to Join() have no effect and return
  // false.
  bool Join();

 private:
  const size_t size_;
  std::queue<std::function<void()> > queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_filled_;
  std::atomic_bool pushing_;
  std::unique_ptr<std::thread[]> threads_;
};

}  // namespace threads
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_THRDPOOL_H_
