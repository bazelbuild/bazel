// Copyright 2021 The Bazel Authors. All rights reserved.
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

#include <IOKit/pwr_mgt/IOPMLib.h>

// Linting disabled for this line because for google code we could use
// absl::Mutex but we cannot yet because Bazel doesn't depend on absl.
#include <mutex>  // NOLINT

#include "src/main/cpp/util/logging.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

// Protects all of the g_sleep_state_* statics.
static std::mutex g_sleep_state_mutex;

// Keep track of our pushes and pops of sleep state.
static int g_sleep_state_stack = 0;

// Our assertion for disabling sleep.
static IOPMAssertionID g_sleep_state_assertion = kIOPMNullAssertionID;

int portable_push_disable_sleep() {
  std::lock_guard<std::mutex> lock(g_sleep_state_mutex);
  BAZEL_CHECK_GE(g_sleep_state_stack, 0);
  if (g_sleep_state_stack == 0) {
    BAZEL_CHECK_EQ(g_sleep_state_assertion, kIOPMNullAssertionID);
    CFStringRef reasonForActivity = CFSTR("build.bazel");
    IOReturn success = IOPMAssertionCreateWithName(
        kIOPMAssertionTypeNoIdleSleep, kIOPMAssertionLevelOn, reasonForActivity,
        &g_sleep_state_assertion);
    BAZEL_CHECK_EQ(success, kIOReturnSuccess);
  }
  g_sleep_state_stack += 1;
  return 0;
}

int portable_pop_disable_sleep() {
  std::lock_guard<std::mutex> lock(g_sleep_state_mutex);
  BAZEL_CHECK_GT(g_sleep_state_stack, 0);
  g_sleep_state_stack -= 1;
  if (g_sleep_state_stack == 0) {
    BAZEL_CHECK_NE(g_sleep_state_assertion, kIOPMNullAssertionID);
    IOReturn success = IOPMAssertionRelease(g_sleep_state_assertion);
    BAZEL_CHECK_EQ(success, kIOReturnSuccess);
    g_sleep_state_assertion = kIOPMNullAssertionID;
  }
  return 0;
}

}  // namespace blaze_jni
