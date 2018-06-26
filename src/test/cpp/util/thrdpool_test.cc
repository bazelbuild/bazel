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

#include <atomic>
#include <thread>  // NOLINT

#include "googletest/include/gtest/gtest.h"
#include "src/main/cpp/util/profiler.h"

namespace blaze_util {
namespace threads {

using blaze_util::profiler::ScopedTask;
using blaze_util::profiler::Task;
using std::atomic_bool;

static void SleepMeasurably(size_t units) {
  // 20ms is long enough for the OS scheduler to preempt the thread.
  std::this_thread::sleep_for(std::chrono::milliseconds(units * 20));
}

TEST(ThreadPoolTest, TestFewerThreadsThanWorkItemsCanCompleteTheWork) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  atomic_bool flag4(false);
  atomic_bool* pflag1 = &flag1;
  atomic_bool* pflag2 = &flag2;
  atomic_bool* pflag3 = &flag3;
  atomic_bool* pflag4 = &flag4;
  EXPECT_TRUE(pool.Push([pflag1]() { *pflag1 = true; }));
  EXPECT_TRUE(pool.Push([pflag2]() { *pflag2 = true; }));
  EXPECT_TRUE(pool.Push([pflag3]() { *pflag3 = true; }));
  EXPECT_TRUE(pool.Push([pflag4]() { *pflag4 = true; }));
  pool.Join();

  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);
  EXPECT_TRUE(flag3);
  EXPECT_TRUE(flag4);
}

TEST(ThreadPoolTest, TestThreadsWorkInParallel) {
  ThreadPool pool(3);

  Task total("total");
  Task task1("t1");
  Task task2("t2");
  Task task3("t3");
  Task* ptask1 = &task1;
  Task* ptask2 = &task2;
  Task* ptask3 = &task3;

  {
    ScopedTask prof(&total);
    EXPECT_TRUE(pool.Push([ptask1]() {
      ScopedTask prof(ptask1);
      SleepMeasurably(1);
    }));
    EXPECT_TRUE(pool.Push([ptask2]() {
      ScopedTask prof(ptask2);
      SleepMeasurably(2);
    }));
    EXPECT_TRUE(pool.Push([ptask3]() {
      ScopedTask prof(ptask3);
      SleepMeasurably(3);
    }));
    pool.Join();
  }

  // Sleep does not guarantee to terminate in a timely manner, so do not assert
  // concrete sleep lengths, nor hard relations -- allow for equality (even if
  // that's very unlikely with a high-precision clock).
  EXPECT_LE(task1.GetDuration().micros_, task2.GetDuration().micros_);
  EXPECT_LE(task2.GetDuration().micros_, task3.GetDuration().micros_);
  EXPECT_LE(task3.GetDuration().micros_, total.GetDuration().micros_);
  EXPECT_LE(total.GetDuration().micros_,
            task3.GetDuration().micros_ + task2.GetDuration().micros_);
}

TEST(ThreadPoolTest, TestCannotPushAfterWorkerThreadsJoined) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  atomic_bool* pflag1 = &flag1;
  atomic_bool* pflag2 = &flag2;
  atomic_bool* pflag3 = &flag3;
  EXPECT_TRUE(pool.Push([pflag1]() { *pflag1 = true; }));
  EXPECT_TRUE(pool.Push([pflag2]() { *pflag2 = true; }));

  pool.Join();
  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);
  EXPECT_FALSE(pool.Push([pflag3]() { *pflag3 = true; }));

  // Join() again has no effect.
  pool.Join();
  EXPECT_FALSE(flag3);
}

}  // namespace threads
}  // namespace blaze_util
