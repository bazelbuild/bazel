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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "src/main/cpp/util/profiler.h"

#include <windows.h>

#include <string>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {
namespace profiler {

Ticks Ticks::Now() {
  LARGE_INTEGER counter;
  if (!QueryPerformanceCounter(&counter)) {
    std::string error = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "QueryPerformanceCounter failed: " << error;
  }
  return {counter.QuadPart};
}

Duration Duration::FromTicks(const Ticks t) {
  LARGE_INTEGER freq;
  if (!QueryPerformanceFrequency(&freq)) {
    std::string error = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "QueryPerformanceFrequency failed: " << error;
  }
  return {(t.value_ * 1000LL * 1000LL) / freq.QuadPart};
}

}  // namespace profiler
}  // namespace blaze_util
