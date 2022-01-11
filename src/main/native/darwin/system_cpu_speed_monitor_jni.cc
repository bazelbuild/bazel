// Copyright 2022 The Bazel Authors. All rights reserved.
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

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPM.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <mach/mach_error.h>
#include <notify.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

void portable_start_cpu_speed_monitoring() {
  // We install a test notification as well that can be used for testing.
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    int token;
    int status = notify_register_dispatch(kIOPMCPUPowerNotificationKey, &token,
                                          queue, ^(int state) {
                                            int value = portable_cpu_speed();
                                            cpu_speed_callback(value);
                                          });
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);

    // This is registered solely so we can test the system from end-to-end.
    // Using the Apple notification requires admin access.
    status = notify_register_dispatch(
        "com.google.bazel.test.cpuspeed", &token, queue, ^(int t) {
          uint64_t state;
          uint32_t status = notify_get_state(t, &state);
          if (status != NOTIFY_STATUS_OK) {
            BAZEL_LOG(FATAL) << "notify_get_state failed: " << status;
          }
          cpu_speed_callback(state);
        });
    BAZEL_CHECK_EQ(status, NOTIFY_STATUS_OK);
  });
}

int portable_cpu_speed() {
  CFDictionaryRef cfCpuPowerStatus = NULL;
  IOReturn ret = IOPMCopyCPUPowerStatus(&cfCpuPowerStatus);
  if (ret == kIOReturnNotFound) {
    // This is a common occurrence when starting up so don't bother logging.
    return -1;
  } else if (ret != kIOReturnSuccess) {
    BAZEL_LOG(ERROR) << "IOPMCopyCPUPowerStatus failed: "
                     << mach_error_string(ret) << "(" << ret << ")";
    return -1;
  }
  CFNumberRef cfSpeed = static_cast<CFNumberRef>(CFDictionaryGetValue(
      cfCpuPowerStatus, CFSTR(kIOPMCPUPowerLimitProcessorSpeedKey)));

  if (!cfSpeed) {
    BAZEL_LOG(ERROR)
        << "IOPMCopyCPUPowerStatus missing kIOPMCPUPowerLimitProcessorSpeedKey";
    CFRelease(cfCpuPowerStatus);
    return -1;
  }
  int speed;
  CFNumberGetValue(cfSpeed, kCFNumberIntType, &speed);
  CFRelease(cfCpuPowerStatus);

  BAZEL_LOG(USER) << "cpu speed anomaly: " << speed;

  return speed;
}

}  // namespace blaze_jni
