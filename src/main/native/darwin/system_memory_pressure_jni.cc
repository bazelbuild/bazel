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

#include <notify.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

void portable_start_memory_pressure_monitoring() {
  // To test use:
  //   sudo memory_pressure -S -l warn
  //   sudo memory_pressure -S -l critical
  // or use the test notifications that we register.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL, queue);
    BAZEL_CHECK_NE(source, nullptr);
    dispatch_source_set_event_handler(source, ^{
      dispatch_source_memorypressure_flags_t pressureLevel =
          dispatch_source_get_data(source);
      if (pressureLevel == DISPATCH_MEMORYPRESSURE_WARN) {
        BAZEL_LOG(USER) << "memory pressure warning anomaly";
        memory_pressure_callback(MemoryPressureLevelWarning);
      } else if (pressureLevel == DISPATCH_MEMORYPRESSURE_CRITICAL) {
        BAZEL_LOG(USER) << "memory pressure critical anomaly";
        memory_pressure_callback(MemoryPressureLevelCritical);
      } else {
        BAZEL_LOG(FATAL) << "unknown memory pressure critical level: "
                         << pressureLevel;
      }
    });
    dispatch_resume(source);
    // These are registered solely so we can test the system from end-to-end.
    // Using the Apple memory_pressure requires admin access.
    int testToken;
    int32_t status = notify_register_dispatch(
        "com.google.bazel.test.memorypressurelevel.warning", &testToken, queue,
        ^(int state) {
          BAZEL_LOG(USER) << "memory pressure test warning anomaly";
          memory_pressure_callback(MemoryPressureLevelWarning);
        });
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
    status = notify_register_dispatch(
        "com.google.bazel.test.memorypressurelevel.critical", &testToken, queue,
        ^(int state) {
          BAZEL_LOG(USER) << "memory pressure test critical anomaly";
          memory_pressure_callback(MemoryPressureLevelCritical);
        });
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
  });
}

}  // namespace blaze_jni
