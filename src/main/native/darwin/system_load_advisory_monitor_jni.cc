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
#include <notify.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

static int gSystemLoadAdvisoryNotifyToken = 0;

void portable_start_system_load_advisory_monitoring() {
  // We install a test notification as well that can be used for testing.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    notify_handler_t handler = (^(int state) {
      int value = portable_system_load_advisory();
      system_load_advisory_callback(value);
    });
    int status = notify_register_dispatch(kIOSystemLoadAdvisoryNotifyName,
                                          &gSystemLoadAdvisoryNotifyToken,
                                          queue, handler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);

    // This is registered solely so we can test the system from end-to-end.
    // Using the Apple notification requires admin access.
    int testToken;
    status = notify_register_dispatch(
        "com.google.bazel.test.SystemLoadAdvisory", &testToken, queue, handler);
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
  });
}

int portable_system_load_advisory() {
  uint64_t state;
  uint32_t status = notify_get_state(gSystemLoadAdvisoryNotifyToken, &state);
  if (status != NOTIFY_STATUS_OK) {
    BAZEL_LOG(FATAL) << "notify_get_state failed:" << status;
  }
  IOSystemLoadAdvisoryLevel advisoryLevel = (IOSystemLoadAdvisoryLevel)state;
  int load = -1;
  switch (advisoryLevel) {
    case kIOSystemLoadAdvisoryLevelGreat:
      BAZEL_LOG(USER) << "system load advisory great (0) anomaly";
      load = 0;
      break;

    case kIOSystemLoadAdvisoryLevelOK:
      BAZEL_LOG(USER) << "system load advisory ok (25) anomaly";
      load = 25;
      break;

    case kIOSystemLoadAdvisoryLevelBad:
      BAZEL_LOG(USER) << "system load advisory bad (75) anomaly";
      load = 75;
      break;
  }
  if (load == -1) {
    BAZEL_LOG(FATAL) << "unknown system load advisory level: " << advisoryLevel;
  }

  return load;
}

}  // namespace blaze_jni
