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

#include <jni.h>
#include <notify.h>

#include <cstdint>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

// These need to be kept in sync with constants in
// j/c/g/devtools/build/lib/platform/SystemMemoryPressureMonitor.java
enum MemoryPressureLevel {
  MemoryPressureLevelNormal = 0,
  MemoryPressureLevelWarning = 1,
  MemoryPressureLevelCritical = 2,
};

static MemoryPressureLevel g_memory_pressure_level = MemoryPressureLevelNormal;
static jobject g_memory_pressure_callback = nullptr;

static void call_memory_pressure_callback(
    dispatch_source_memorypressure_flags_t pressure_level) {
  if (pressure_level == DISPATCH_MEMORYPRESSURE_NORMAL) {
    BAZEL_LOG(USER) << "memory pressure normal anomaly";
    g_memory_pressure_level = MemoryPressureLevelNormal;
  } else if (pressure_level == DISPATCH_MEMORYPRESSURE_WARN) {
    BAZEL_LOG(USER) << "memory pressure warning anomaly";
    g_memory_pressure_level = MemoryPressureLevelWarning;
  } else if (pressure_level == DISPATCH_MEMORYPRESSURE_CRITICAL) {
    BAZEL_LOG(USER) << "memory pressure critical anomaly";
    g_memory_pressure_level = MemoryPressureLevelCritical;
  } else {
    BAZEL_LOG(FATAL) << "unknown memory pressure level: " << pressure_level;
  }
  blaze_jni::PerformIntegerValueCallback(g_memory_pressure_callback,
                                         g_memory_pressure_level);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_platform_SystemMemoryPressureMonitor_registerMemoryPressureNative(
    JNIEnv* env, jclass, jobject callback) {
  if (!blaze_jni::RegisterIntConsumerCallback(
          env, callback, &g_memory_pressure_callback, "memory pressure")) {
    return;
  }
  // To test use:
  //   sudo memory_pressure -S -l warn
  //   sudo memory_pressure -S -l critical
  // or use the test notifications that we register.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
            DISPATCH_MEMORYPRESSURE_CRITICAL,
        queue);
    if (source != nullptr) {
      dispatch_source_set_event_handler(source, ^{
        dispatch_source_memorypressure_flags_t pressure_level =
            dispatch_source_get_data(source);
        call_memory_pressure_callback(pressure_level);
      });
      dispatch_resume(source);
    } else {
      BAZEL_LOG(ERROR) << "dispatch_source_create for memory pressure failed";
    }
    // These are registered solely so we can test the system from end-to-end.
    // Using the Apple memory_pressure requires admin access.
    int testToken;
    int32_t status = notify_register_dispatch(
        "com.google.bazel.test.memorypressurelevel", &testToken, queue,
        ^(int token) {
          uint64_t state;
          uint32_t status = notify_get_state(token, &state);
          if (status != NOTIFY_STATUS_OK) {
            BAZEL_LOG(ERROR) << "notify_get_state failed: " << status;
            return;
          }
          dispatch_source_memorypressure_flags_t pressure_level;
          switch (state) {
            case 0:
              pressure_level = DISPATCH_MEMORYPRESSURE_NORMAL;
              break;
            case 1:
              pressure_level = DISPATCH_MEMORYPRESSURE_WARN;
              break;
            case 2:
              pressure_level = DISPATCH_MEMORYPRESSURE_CRITICAL;
              break;
            default:
              BAZEL_LOG(ERROR) << "unknown memory pressure level: " << state;
              pressure_level = -1;
              break;
          }
          if (pressure_level != -1) {
            call_memory_pressure_callback(pressure_level);
          }
        });
    if (status != NOTIFY_STATUS_OK) {
      BAZEL_LOG(ERROR) << "notify_register_dispatch failed: " << status;
    }
  });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_platform_SystemMemoryPressureMonitor_systemMemoryPressureNative(
    JNIEnv*, jclass) {
  return g_memory_pressure_level;
}

}  // namespace blaze_jni
