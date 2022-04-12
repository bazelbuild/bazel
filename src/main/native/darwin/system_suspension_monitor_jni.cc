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

#include <IOKit/IOMessage.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <notify.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/darwin/util.h"
#include "src/main/native/unix_jni.h"

namespace blaze_jni {

typedef struct {
  // Port used to relay sleep call back messages.
  io_connect_t connect_port;
} SuspendState;

static void SleepCallBack(void *refcon, io_service_t service,
                          natural_t message_type, void *message_argument) {
  SuspendState *state = (SuspendState *)refcon;
  switch (message_type) {
    case kIOMessageCanSystemSleep:
      // This needs to be handled to allow sleep.
      IOAllowPowerChange(state->connect_port, (intptr_t)message_argument);
      break;

    case kIOMessageSystemWillSleep:
      BAZEL_LOG(USER) << "suspend anomaly due to kIOMessageSystemWillSleep";
      suspend_callback(SuspensionReasonSleep);
      // This needs to be acknowledged to allow sleep.
      IOAllowPowerChange(state->connect_port, (intptr_t)message_argument);
      break;

    case kIOMessageSystemHasPoweredOn:
      // Note that Macs have a "Dark Wake" mode (also known as PowerNap) which
      // can have the processors (and disk and network) turn on.
      // (https://support.apple.com/en-us/HT204032). Dark Wake does NOT
      // trigger PowerOn messages through our sleep callbacks, but can allow
      // builds to proceed for a considerable amount of time (for example if
      // Time Machine is performing a back up).
      // There is currently a race condition where a build may finish
      // between the time we receive the kIOMessageSystemWillSleep and the
      // machine actually goes to sleep (roughly 20 seconds in my experiments).
      // This will result in us reporting that the build was suspended when it
      // wasn't. I haven't come up with an smart way of avoiding this issue, but
      // I don't think we really care. Over reporting "suspensions" is better
      // than under reporting them.
      BAZEL_LOG(USER) << "suspend anomaly due to kIOMessageSystemHasPoweredOn";
      suspend_callback(SuspensionReasonWake);
      break;

    case kIOMessageSystemWillPowerOn:
    case kIOMessageSystemWillNotSleep:
      // We don't handle will not sleep. This can only occur is somebody else
      // cancels the sleep, and will never occur AFTER a
      // kIOMessageSystemWillSleep.
      // We don't handle will power on, we only care when it HAS powered on.
    default:
      break;
  }
}

void portable_start_suspend_monitoring() {
  static dispatch_once_t once_token;
  static SuspendState suspend_state;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = bazel::darwin::JniDispatchQueue();
    IONotificationPortRef notifyPortRef;
    io_object_t notifierObject;

    // Register to receive system sleep notifications.
    // Testing needs to be done manually. Use the logging to verify
    // that sleeps are being caught here.
    suspend_state.connect_port = IORegisterForSystemPower(
        &suspend_state, &notifyPortRef, SleepCallBack, &notifierObject);
    BAZEL_CHECK_NE(suspend_state.connect_port, MACH_PORT_NULL);
    IONotificationPortSetDispatchQueue(notifyPortRef, queue);

    // Register to deal with SIGCONT.
    // We register for SIGCONT because we can't catch SIGSTOP.
    // We do have the potential of "over counting" suspensions if you send
    // multiple SIGCONTs to a process without a previous SIGSTOP/SIGTSTP,
    // but there is no reason to send a SIGCONT without a SIGSTOP/SIGTSTP, and
    // having this functionality gives us some ability to unit test suspension
    // counts.
    sig_t signal_val = signal(SIGCONT, SIG_IGN);
    BAZEL_CHECK_NE(signal_val, SIG_ERR);
    dispatch_source_t signal_source =
        dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGCONT, 0, queue);
    BAZEL_CHECK_NE(signal_source, nullptr);
    dispatch_source_set_event_handler(signal_source, ^{
      BAZEL_LOG(USER) << "suspend anomaly due to SIGCONT";
      suspend_callback(SuspensionReasonSIGCONT);
    });
    dispatch_resume(signal_source);
    signal_source =
        dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGTSTP, 0, queue);
    BAZEL_CHECK_NE(signal_source, nullptr);
    dispatch_source_set_event_handler(signal_source, ^{
      BAZEL_LOG(USER) << "suspend anomaly due to SIGTSTP";
      suspend_callback(SuspensionReasonSIGTSTP);
    });
    dispatch_resume(signal_source);
  });
}

}  // namespace blaze_jni
