// Copyright 2014 The Bazel Authors. All rights reserved.
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

#include "src/main/native/unix_jni.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <IOKit/IOMessage.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <os/log.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <sys/types.h>
#include <sys/xattr.h>

#include <atomic>
// Linting disabled for this line because for google code we could use
// absl::Mutex but we cannot yet because Bazel doesn't depend on absl.
#include <mutex>  // NOLINT
#include <string>

namespace blaze_jni {

const int PATH_MAX2 = PATH_MAX * 2;

using std::string;

// See unix_jni.h.
string ErrorMessage(int error_number) {
  char buf[1024] = "";
  if (strerror_r(error_number, buf, sizeof buf) < 0) {
    snprintf(buf, sizeof buf, "strerror_r(%d): errno %d", error_number, errno);
  }

  return string(buf);
}


int portable_fstatat(
    int dirfd, char *name, portable_stat_struct *statbuf, int flags) {
  char dirPath[PATH_MAX2];  // Have enough room for relative path

  // No fstatat under darwin, simulate it
  if (flags != 0) {
    // We don't support any flags
    errno = ENOSYS;
    return -1;
  }
  if (strlen(name) == 0 || name[0] == '/') {
    // Absolute path, simply stat
    return portable_stat(name, statbuf);
  }
  // Relative path, construct an absolute path
  if (fcntl(dirfd, F_GETPATH, dirPath) == -1) {
    return -1;
  }
  int l = strlen(dirPath);
  if (dirPath[l-1] != '/') {
    // dirPath is twice the PATH_MAX size, we always have room for the extra /
    dirPath[l] = '/';
    dirPath[l+1] = 0;
    l++;
  }
  strncat(dirPath, name, PATH_MAX2-l-1);
  char *newpath = realpath(dirPath, nullptr);  // this resolve the relative path
  if (newpath == nullptr) {
    return -1;
  }
  int r = portable_stat(newpath, statbuf);
  free(newpath);
  return r;
}

int StatSeconds(const portable_stat_struct &statbuf, StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atime;
    case STAT_CTIME:
      return statbuf.st_ctime;
    case STAT_MTIME:
      return statbuf.st_mtime;
    default:
      CHECK(false);
  }
}

int StatNanoSeconds(const portable_stat_struct &statbuf, StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atimespec.tv_nsec;
    case STAT_CTIME:
      return statbuf.st_ctimespec.tv_nsec;
    case STAT_MTIME:
      return statbuf.st_mtimespec.tv_nsec;
    default:
      CHECK(false);
  }
}

ssize_t portable_getxattr(const char *path, const char *name, void *value,
                          size_t size, bool *attr_not_found) {
  ssize_t result = getxattr(path, name, value, size, 0, 0);
  *attr_not_found = (errno == ENOATTR);
  return result;
}

ssize_t portable_lgetxattr(const char *path, const char *name, void *value,
                           size_t size, bool *attr_not_found) {
  ssize_t result = getxattr(path, name, value, size, 0, XATTR_NOFOLLOW);
  *attr_not_found = (errno == ENOATTR);
  return result;
}

int portable_sysctlbyname(const char *name_chars, long *mibp, size_t *sizep) {
  return sysctlbyname(name_chars, mibp, sizep, nullptr, 0);
}

// Queue used for all of our anomaly tracking.
static dispatch_queue_t JniDispatchQueue() {
  static dispatch_once_t once_token;
  static dispatch_queue_t queue;
  dispatch_once(&once_token, ^{
    queue = dispatch_queue_create("build.bazel.jni", DISPATCH_QUEUE_SERIAL);
    CHECK(queue);
  });
  return queue;
}

// Log used for all of our anomaly logging.
// Logging can be traced using:
// `log stream -level debug --predicate '(subsystem == "build.bazel")'`
//
// This may return NULL if `os_log_create` is not supported on this version of
// macOS. Use `log_if_possible` to log when supported.
static os_log_t JniOSLog() {
  static dispatch_once_t once_token;
  static os_log_t log = nullptr;
  // On macOS < 10.12, os_log_create is not available. Since we target 10.10,
  // this will be weakly linked and can be checked for availability at run
  // time.
  if (&os_log_create != nullptr) {
    dispatch_once(&once_token, ^{
      log = os_log_create("build.bazel", "jni");
      CHECK(log);
    });
  }
  return log;
}

// The macOS implementation asserts that `msg` be a string literal (not just a
// const char*), so we cannot use a function.
#define log_if_possible(msg)   \
  do {                         \
    os_log_t log = JniOSLog(); \
    if (log != nullptr) {      \
      os_log_debug(log, msg);  \
    }                          \
  } while (0);

// Protects all of the g_sleep_state_* statics.
// value is "leaked" intentionally because std::mutex is not trivially
// destructible at this time, g_sleep_state_mutex is a singleton, and
// leaking it has no consequences.
static std::mutex *g_sleep_state_mutex = new std::mutex;

// Keep track of our pushes and pops of sleep state.
static int g_sleep_state_stack = 0;

// Our assertion for disabling sleep.
static IOPMAssertionID g_sleep_state_assertion = kIOPMNullAssertionID;

int portable_push_disable_sleep() {
  std::lock_guard<std::mutex> lock(*g_sleep_state_mutex);
  assert(g_sleep_state_stack >= 0);
  if (g_sleep_state_stack == 0) {
    assert(g_sleep_state_assertion == kIOPMNullAssertionID);
    CFStringRef reasonForActivity = CFSTR("build.bazel");

    IOReturn success = IOPMAssertionCreateWithName(
        kIOPMAssertionTypeNoIdleSleep, kIOPMAssertionLevelOn, reasonForActivity,
        &g_sleep_state_assertion);
    assert(success == kIOReturnSuccess);
    log_if_possible("sleep assertion created");
  }
  g_sleep_state_stack += 1;
  return 0;
}

int portable_pop_disable_sleep() {
  std::lock_guard<std::mutex> lock(*g_sleep_state_mutex);
  assert(g_sleep_state_stack > 0);
  g_sleep_state_stack -= 1;
  if (g_sleep_state_stack == 0) {
    assert(g_sleep_state_assertion != kIOPMNullAssertionID);
    IOReturn success = IOPMAssertionRelease(g_sleep_state_assertion);
    assert(success == kIOReturnSuccess);
    g_sleep_state_assertion = kIOPMNullAssertionID;
    log_if_possible("sleep assertion released");
  }
  return 0;
}

typedef struct {
  // Port used to relay sleep call back messages.
  io_connect_t connect_port;

  // Count of suspensions. Atomic because it can be read from any java thread
  // and is written to from a dispatch_queue thread.
  std::atomic_int suspend_count;
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
      log_if_possible("suspend anomaly due to kIOMessageSystemWillSleep");
      ++state->suspend_count;
      // This needs to be acknowledged to allow sleep.
      IOAllowPowerChange(state->connect_port, (intptr_t)message_argument);
      break;

    case kIOMessageSystemWillNotSleep:
      log_if_possible(
          "suspend anomaly cancelled due to kIOMessageSystemWillNotSleep");
      --state->suspend_count;
      break;

    case kIOMessageSystemWillPowerOn:
    case kIOMessageSystemHasPoweredOn:
      // We increment g_suspend_count when we are alerted to the sleep as
      // opposed to when we wake up, because Macs have a "Dark Wake" mode (also
      // known as PowerNap) which is when the processors (and disk and network)
      // turn on for brief periods of time
      // (https://support.apple.com/en-us/HT204032). Dark Wake does NOT trigger
      // PowerOn messages through our sleep callbacks, but can allow
      // builds to proceed for a considerable amount of time (for example if
      // Time Machine is performing a back up).
      // There is currently a race condition where a build may finish
      // between the time we receive the kIOMessageSystemWillSleep and the
      // machine actually goes to sleep (roughly 20 seconds in my experiments)
      // or between the time we receive the kIOMessageSystemWillSleep and
      // kIOMessageSystemWillNotSleep. This will result in us reporting that the
      // build was suspended when it wasn't. I haven't come up with an smart way
      // of avoiding this issue, but I don't think we really care. Over
      // reporting "suspensions" is better than under reporting them.
    default:
      break;
  }
}

int portable_suspend_count() {
  static dispatch_once_t once_token;
  static SuspendState suspend_state;
  dispatch_once(&once_token, ^{
    dispatch_queue_t queue = JniDispatchQueue();
    IONotificationPortRef notifyPortRef;
    io_object_t notifierObject;

    // Register to receive system sleep notifications.
    // Testing needs to be done manually. Use the logging to verify
    // that sleeps are being caught here.
    // `log stream -level debug --predicate '(subsystem == "build.bazel")'`
    suspend_state.connect_port = IORegisterForSystemPower(
        &suspend_state, &notifyPortRef, SleepCallBack, &notifierObject);
    CHECK(suspend_state.connect_port != MACH_PORT_NULL);
    IONotificationPortSetDispatchQueue(notifyPortRef, queue);

    // Register to deal with SIGCONT.
    // We register for SIGCONT because we can't catch SIGSTOP and we can't
    // distinguish a SIGCONT after a SIGSTOP from a SIGCONT after SIGTSTP.
    // We do have the potential of "over counting" suspensions if you send
    // multiple SIGCONTs to a process without a previous SIGSTOP/SIGTSTP,
    // but there is no reason to send a SIGCONT without a SIGSTOP/SIGTSTP, and
    // having this functionality gives us some ability to unit test suspension
    // counts.
    sig_t signal_val = signal(SIGCONT, SIG_IGN);
    CHECK(signal_val != SIG_ERR);
    dispatch_source_t signal_source =
        dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGCONT, 0, queue);
    CHECK(signal_source != nullptr);
    dispatch_source_set_event_handler(signal_source, ^{
      log_if_possible("suspend anomaly due to SIGCONT");
      ++suspend_state.suspend_count;
    });
    dispatch_resume(signal_source);
  });
  return suspend_state.suspend_count;
}

int portable_memory_pressure_warning_count() {
  // To test use:
  // log stream -level debug --predicate '(subsystem == "build.bazel")'
  // sudo memory_pressure -S -l warn
  static dispatch_once_t once_token;
  static std::atomic_int warning_count;
  dispatch_once(&once_token, ^{
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, DISPATCH_MEMORYPRESSURE_WARN,
        JniDispatchQueue());
    CHECK(source != nullptr);
    dispatch_source_set_event_handler(source, ^{
      dispatch_source_memorypressure_flags_t pressureLevel =
          dispatch_source_get_data(source);
      if (pressureLevel == DISPATCH_MEMORYPRESSURE_WARN) {
        log_if_possible("memory pressure warning anomaly");
        ++warning_count;
      }
    });
    dispatch_resume(source);
  });
  return warning_count;
}

int portable_memory_pressure_critical_count() {
  // To test use:
  // log stream -level debug --predicate '(subsystem == "build.bazel")'
  // sudo memory_pressure -S -l critical
  static dispatch_once_t once_token;
  static std::atomic_int critical_count;
  dispatch_once(&once_token, ^{
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_CRITICAL, JniDispatchQueue());
    CHECK(source != nullptr);
    dispatch_source_set_event_handler(source, ^{
      dispatch_source_memorypressure_flags_t pressureLevel =
          dispatch_source_get_data(source);
      if (pressureLevel == DISPATCH_MEMORYPRESSURE_CRITICAL) {
        log_if_possible("memory pressure critical anomaly");
        ++critical_count;
      }
    });
    dispatch_resume(source);
  });
  return critical_count;
}

}  // namespace blaze_jni
