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

#include <thread>  // NOLINT

#include "googletest/include/gtest/gtest.h"

namespace blaze_util {
namespace profiler {

static void SleepMeasurably() {
  // The profiler should have at least 1 ms precision.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

TEST(ProfilerTest, TestStopWatchMeasuresElapsedTime) {
  StopWatch sw1, sw2;

  SleepMeasurably();
  Duration t1 = Duration::FromTicks(sw1.Elapsed());
  SleepMeasurably();
  Duration t2 = Duration::FromTicks(sw1.Elapsed());

  // Assert that two sleeps show a longer elapsed time than one sleep.
  ASSERT_GT(t2.micros_, t1.micros_);

  sw2.Reset();
  SleepMeasurably();
  Duration t3_not_reset = Duration::FromTicks(sw1.Elapsed());
  Duration t3_reset = Duration::FromTicks(sw2.Elapsed());

  // Assert that sleeping the same amounts, a Reset() call results in less
  // elapsed time in one StopWatch than in the other. (This way we don't rely on
  // sleep completing in a timely manner.)
  ASSERT_GT(t3_not_reset.micros_, t3_reset.micros_);
}

TEST(ProfilerTest, TestScopedTaskMeasuresElapsedTime) {
  Task scope1("task 2"), scope2("task 2"), scope_both("tasks 1 and 2");
  {
    ScopedTask p1(&scope1), p2(&scope_both);
    SleepMeasurably();
  }
  {
    ScopedTask p1(&scope2), p2(&scope_both);
    SleepMeasurably();
    SleepMeasurably();
  }
  ASSERT_GT(scope_both.GetDuration().micros_, scope1.GetDuration().micros_);
  ASSERT_GT(scope_both.GetDuration().micros_, scope2.GetDuration().micros_);
  ASSERT_EQ(scope1.GetCalls(), 1u);
  ASSERT_EQ(scope2.GetCalls(), 1u);
  ASSERT_EQ(scope_both.GetCalls(), 2u);
}

}  // namespace profiler
}  // namespace blaze_util
