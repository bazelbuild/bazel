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
#include <notify_keys.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

// Not defined by Apple headers, but definitely sent out with macOS 12.
// Named the same as what we would expect from Apple so that hopefully if/when
// they make it public the compiler will let us know.
const char *kNotifyVFSVeryLowDiskSpace = "com.apple.system.verylowdiskspace";

namespace blaze_jni {

void portable_start_disk_space_monitoring() {
  // We install a test notification as well that can be used for testing.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    notify_handler_t lowHandler = (^(int token) {
      BAZEL_LOG(USER) << "disk space low anomaly";
      disk_space_callback(DiskSpaceLevelLow);
    });
    notify_handler_t veryLowHandler = (^(int token) {
      BAZEL_LOG(USER) << "disk space very low anomaly";
      disk_space_callback(DiskSpaceLevelVeryLow);
    });
    int notifyToken;
    int status = notify_register_dispatch(kNotifyVFSLowDiskSpace,
                                          &notifyToken,
                                          queue, lowHandler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
    status = notify_register_dispatch(kNotifyVFSVeryLowDiskSpace,
                                          &notifyToken,
                                          queue, veryLowHandler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
    // These are registered solely so we can test the system from end-to-end.
    status = notify_register_dispatch(
        "com.google.bazel.test.diskspace.low", &notifyToken, queue, lowHandler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
    status = notify_register_dispatch("com.google.bazel.test.diskspace.verylow",
                                      &notifyToken, queue, veryLowHandler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
  });
}

}  // namespace blaze_jni
