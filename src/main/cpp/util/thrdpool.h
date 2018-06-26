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
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace blaze_util {
namespace threads {

class ThreadPool {
 public:
  ThreadPool(const size_t size);
  ~ThreadPool();
  bool Push(std::function<void()> func);
  void Join();

 private:
  const size_t size_;
  std::queue<std::function<void()> > queue_;
  std::mutex queue_mutex_;
  std::atomic_bool pushing_;
  std::unique_ptr<std::thread[]> threads_;
};

}  // namespace threads
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_THRDPOOL_H_
