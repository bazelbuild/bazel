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
#include <notify_keys.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

// Not defined by Apple headers, but definitely sent out with macOS 12.
// Named the same as what we would expect from Apple so that hopefully if/when
// they make it public the compiler will let us know.
const char *kNotifyVFSVeryLowDiskSpace = "com.apple.system.verylowdiskspace";

namespace blaze_jni {

// These need to be kept in sync with constants in
// j/c/g/devtools/build/lib/platform/SystemDiskSpaceEvent.java
enum DiskSpaceLevel {
  DiskSpaceLevelLow = 0,
  DiskSpaceLevelVeryLow = 1,
};

static jobject g_disk_space_callback = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_platform_PlatformNativeDepsServiceImpl_registerDiskSpaceNative(
    JNIEnv* env, jobject, jobject callback) {
  if (!blaze_jni::RegisterIntConsumerCallback(
          env, callback, &g_disk_space_callback, "disk space")) {
    return;
  }
  // We install a test notification as well that can be used for testing.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    notify_handler_t lowHandler = (^(int token) {
      BAZEL_LOG(USER) << "disk space low anomaly";
      blaze_jni::PerformIntegerValueCallback(g_disk_space_callback,
                                             DiskSpaceLevelLow);
    });
    notify_handler_t veryLowHandler = (^(int token) {
      BAZEL_LOG(USER) << "disk space very low anomaly";
      blaze_jni::PerformIntegerValueCallback(g_disk_space_callback,
                                             DiskSpaceLevelVeryLow);
    });
    int notifyToken;
    int status = notify_register_dispatch(kNotifyVFSLowDiskSpace,
                                          &notifyToken,
                                          queue, lowHandler);
    if (status != NOTIFY_STATUS_OK) {
      BAZEL_LOG(ERROR) << "notify_register_dispatch failed: " << status;
    }
    status = notify_register_dispatch(kNotifyVFSVeryLowDiskSpace,
                                          &notifyToken,
                                          queue, veryLowHandler);
    if (status != NOTIFY_STATUS_OK) {
      BAZEL_LOG(ERROR) << "notify_register_dispatch failed: " << status;
    }
    // These are registered solely so we can test the system from end-to-end.
    status = notify_register_dispatch(
        "com.google.bazel.test.diskspace.low", &notifyToken, queue, lowHandler);
    if (status != NOTIFY_STATUS_OK) {
      BAZEL_LOG(ERROR) << "notify_register_dispatch failed: " << status;
    }
    status = notify_register_dispatch("com.google.bazel.test.diskspace.verylow",
                                      &notifyToken, queue, veryLowHandler);
    if (status != NOTIFY_STATUS_OK) {
      BAZEL_LOG(ERROR) << "notify_register_dispatch failed: " << status;
    }
  });
}

}  // namespace blaze_jni
