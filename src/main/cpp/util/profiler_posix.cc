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

#include "src/main/cpp/util/profiler.h"

#include <time.h>

#include <string>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {
namespace profiler {

Ticks Ticks::Now() {
  struct timespec ts = {};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
    std::string error = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "clock_gettime failed: " << error;
  }
  return {ts.tv_sec * 1000LL * 1000LL * 1000LL + ts.tv_nsec};
}

Duration Duration::FromTicks(const Ticks ticks) {
  return {ticks.value_ / 1000LL};
}

}  // namespace profiler
}  // namespace blaze_util
