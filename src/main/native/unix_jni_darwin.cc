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
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <sys/types.h>
#include <sys/xattr.h>

// Linting disabled for this line because for google code we could use
// absl::Mutex but we cannot yet because Bazel doesn't depend on absl.
#include <mutex>  // NOLINT
#include <string>

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
  char *newpath = realpath(dirPath, NULL);  // this resolve the relative path
  if (newpath == NULL) {
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
  return sysctlbyname(name_chars, mibp, sizep, NULL, 0);
}

// Protects all of the g_sleep_state_* statics.
// value is "leaked" intentionally because std::mutex is not trivially
// destructible at this time, g_sleep_state_mutex is a singleton, and
// leaking it has no consequences.
std::mutex *g_sleep_state_mutex = new std::mutex;

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
  }
  return 0;
}
