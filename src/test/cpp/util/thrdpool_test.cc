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
using std::thread;

static void SleepMeasurably(size_t units) {
  // 20ms is long enough for the OS scheduler to preempt the thread.
  std::this_thread::sleep_for(std::chrono::milliseconds(units * 20));
}

TEST(ThreadPoolTest, TestPoolThatNeverGotWorkPushed) {
  ThreadPool pool(2);
  SleepMeasurably(1);
  EXPECT_TRUE(pool.Join());
}

TEST(ThreadPoolTest, TestFewerThreadsThanWorkItemsCanCompleteTheWork) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  atomic_bool flag4(false);
  EXPECT_TRUE(pool.Push([&flag1]() { flag1 = true; }));
  EXPECT_TRUE(pool.Push([&flag2]() { flag2 = true; }));
  EXPECT_TRUE(pool.Push([&flag3]() { flag3 = true; }));
  EXPECT_TRUE(pool.Push([&flag4]() { flag4 = true; }));
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

  {
    ScopedTask prof(&total);
    EXPECT_TRUE(pool.Push([&task1]() {
      ScopedTask prof(&task1);
      SleepMeasurably(1);
    }));
    EXPECT_TRUE(pool.Push([&task2]() {
      ScopedTask prof(&task2);
      SleepMeasurably(2);
    }));
    EXPECT_TRUE(pool.Push([&task3]() {
      ScopedTask prof(&task3);
      SleepMeasurably(3);
    }));
    pool.Join();
  }

  EXPECT_EQ(task1.GetCalls(), 1u);
  EXPECT_EQ(task2.GetCalls(), 1u);
  EXPECT_EQ(task3.GetCalls(), 1u);
  EXPECT_EQ(total.GetCalls(), 1u);

  // Sleep does not guarantee to terminate in a timely manner, so do not assert
  // concrete sleep lengths, nor hard relations -- allow for equality (even if
  // that's very unlikely with a high-precision clock).
  EXPECT_LE(task1.GetDuration().micros_, task2.GetDuration().micros_);
  EXPECT_LE(task2.GetDuration().micros_, task3.GetDuration().micros_);
  EXPECT_LE(task3.GetDuration().micros_, total.GetDuration().micros_);
  EXPECT_LE(total.GetDuration().micros_,
            task3.GetDuration().micros_ + task2.GetDuration().micros_);
}

TEST(ThreadPoolTest, TestSlowPushFastPop) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  atomic_bool flag4(false);
  EXPECT_TRUE(pool.Push([&flag1]() { flag1 = true; }));
  SleepMeasurably(1);
  EXPECT_TRUE(pool.Push([&flag2]() { flag2 = true; }));
  SleepMeasurably(1);
  EXPECT_TRUE(pool.Push([&flag3]() { flag3 = true; }));
  SleepMeasurably(1);
  EXPECT_TRUE(pool.Push([&flag4]() { flag4 = true; }));
  SleepMeasurably(1);
  pool.Join();

  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);
  EXPECT_TRUE(flag3);
  EXPECT_TRUE(flag4);
}

TEST(ThreadPoolTest, TestFastPushSlowPop) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  atomic_bool flag4(false);
  EXPECT_TRUE(pool.Push([&flag1]() {
    SleepMeasurably(1);
    flag1 = true;
  }));
  EXPECT_TRUE(pool.Push([&flag2]() {
    SleepMeasurably(1);
    flag2 = true;
  }));
  EXPECT_TRUE(pool.Push([&flag3]() {
    SleepMeasurably(1);
    flag3 = true;
  }));
  EXPECT_TRUE(pool.Push([&flag4]() {
    SleepMeasurably(1);
    flag4 = true;
  }));
  pool.Join();

  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);
  EXPECT_TRUE(flag3);
  EXPECT_TRUE(flag4);
}

TEST(ThreadPoolTest, TestCannotPushAfterWorkerThreadsJoined) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);
  atomic_bool flag3(false);
  EXPECT_TRUE(pool.Push([&flag1]() { flag1 = true; }));
  EXPECT_TRUE(pool.Push([&flag2]() { flag2 = true; }));

  pool.Join();
  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);
  EXPECT_FALSE(pool.Push([&flag3]() { flag3 = true; }));

  // Join() again has no effect.
  pool.Join();
  EXPECT_FALSE(flag3);
}

TEST(ThreadPoolTest, TestMultipleThreadsPushingAndJoining) {
  ThreadPool pool(2);

  atomic_bool flag1(false);
  atomic_bool flag2(false);

  // Main thread may Push().
  EXPECT_TRUE(pool.Push([&flag1]() { flag1 = true; }));

  // Another thread may Push().
  atomic_bool success(false);
  thread t1([&pool, &success, &flag2]() {
    success = pool.Push([&flag2]() { flag2 = true; });
  });
  t1.join();
  EXPECT_TRUE(success);

  // Another thread may Join().
  success = false;
  thread t2([&pool, &success]() { success = pool.Join(); });
  t2.join();
  EXPECT_TRUE(success);
  EXPECT_TRUE(flag1);
  EXPECT_TRUE(flag2);

  // Cannot Join() again from main thread.
  EXPECT_FALSE(pool.Join());
}

}  // namespace threads
}  // namespace blaze_util
