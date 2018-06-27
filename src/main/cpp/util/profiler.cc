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

#include <inttypes.h>
#include <stdio.h>

namespace blaze_util {
namespace profiler {

Task::~Task() {
  Duration duration = GetDuration();
  fprintf(stderr, "Task(%s): %" PRIu64 " calls (%" PRId64 " microseconds)\n",
          name_, GetCalls(), duration.micros_);
}

void StopWatch::PrintAndReset(const char* name) {
  Duration elapsed = ElapsedTime();
  fprintf(stderr, "StopWatch(%s): %" PRId64 " microseconds elapsed\n", name,
          elapsed.micros_);
  Reset();
}

}  // namespace profiler
}  // namespace blaze_util
