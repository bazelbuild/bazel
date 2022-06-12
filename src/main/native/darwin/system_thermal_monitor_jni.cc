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

#include <TargetConditionals.h>
#include <libkern/OSThermalNotification.h>
#include <notify.h>

#include "src/main/native/darwin/util.h"
#include "src/main/native/macros.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

static int gThermalNotifyToken = 0;

void portable_start_thermal_monitoring() {
  // To test use:
  //   /usr/bin/log stream -level debug \
  //       --predicate '(subsystem == "build.bazel")'
  //   sudo thermal simulate cpu {nominal|moderate|heavy|trapping|sleeping}
  // Note that we install the test notification as well that can be used for
  // testing.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    notify_handler_t handler = (^(int state) {
      int value = portable_thermal_load();
      thermal_callback(value);
    });
    int status =
        notify_register_dispatch(kOSThermalNotificationPressureLevelName,
                                 &gThermalNotifyToken, queue, handler);
    CHECK(status == NOTIFY_STATUS_OK);

    // This is registered solely so we can test the system from end-to-end.
    // Using the Apple notification requires admin access.
    int testToken;
    status =
        notify_register_dispatch("com.google.bazel.test.thermalpressurelevel",
                                 &testToken, queue, handler);
    CHECK(status == NOTIFY_STATUS_OK);
    log_if_possible("thermal monitoring registered");
  });
}

int portable_thermal_load() {
  uint64_t state;
  uint32_t status = notify_get_state(gThermalNotifyToken, &state);
  if (status != NOTIFY_STATUS_OK) {
    log_if_possible("error: notify_get_state failed (%d)", status);
    return -1;
  }
  OSThermalPressureLevel thermalLevel = (OSThermalPressureLevel)state;
  int load = -1;
  switch (thermalLevel) {
    case kOSThermalPressureLevelNominal:
      log_if_possible("thermal pressure nominal (0) anomaly");
      load = 0;
      break;

    case kOSThermalPressureLevelModerate:
      log_if_possible("thermal pressure moderate (33) anomaly ");
      load = 33;
      break;

    case kOSThermalPressureLevelHeavy:
      log_if_possible("thermal pressure heavy (50) anomaly");
      load = 50;
      break;

    case kOSThermalPressureLevelTrapping:
      log_if_possible("thermal pressure trapping (90) anomaly");
      load = 90;
      break;

    case kOSThermalPressureLevelSleeping:
      log_if_possible("thermal pressure sleeping (100) anomaly");
      load = 100;
      break;
  }
  if (load == -1) {
    log_if_possible("error: unknown thermal pressure level: %d",
                    (int)thermalLevel);
  }

  return load;
}

}  // namespace blaze_jni
