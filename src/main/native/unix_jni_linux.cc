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

#include <errno.h>
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>

#include <string>

#include "src/main/native/unix_jni.h"

namespace blaze_jni {

std::string ErrorMessage(int error_number) {
  char buf[1024] = "";

  // In its infinite wisdom, GNU libc defines strerror_r with extended
  // functionality which is not compatible with not the
  // SUSv3-conformant one which returns an error code; see DESCRIPTION
// at strerror(3).
#if !__GLIBC__ || (_POSIX_C_SOURCE >= 200112L && !_GNU_SOURCE)
  if (strerror_r(error_number, buf, sizeof buf) == -1) {
    return std::string("");
  } else {
    return std::string(buf);
  }
#else
  return std::string(strerror_r(error_number, buf, sizeof buf));
#endif
}

int portable_fstatat(
    int dirfd, char *name, portable_stat_struct *statbuf, int flags) {
  return fstatat64(dirfd, name, statbuf, flags);
}

uint64_t StatEpochMilliseconds(const portable_stat_struct &statbuf,
                               StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atim.tv_sec * 1000L + statbuf.st_atim.tv_nsec / 1000000;
    case STAT_CTIME:
      return statbuf.st_ctim.tv_sec * 1000L + statbuf.st_ctim.tv_nsec / 1000000;
    case STAT_MTIME:
      return statbuf.st_mtim.tv_sec * 1000L + statbuf.st_mtim.tv_nsec / 1000000;
  }
}

ssize_t portable_getxattr(const char *path, const char *name, void *value,
                          size_t size, bool *attr_not_found) {
  ssize_t result = ::getxattr(path, name, value, size);
  *attr_not_found = (errno == ENODATA);
  return result;
}

ssize_t portable_lgetxattr(const char *path, const char *name, void *value,
                           size_t size, bool *attr_not_found) {
  ssize_t result = ::lgetxattr(path, name, value, size);
  *attr_not_found = (errno == ENODATA);
  return result;
}

int portable_push_disable_sleep() {
  // Currently not supported.
  return -1;
}

int portable_pop_disable_sleep() {
  // Currently not supported.
  return -1;
}

void portable_start_suspend_monitoring() {
  // Currently not implemented.
}

void portable_start_thermal_monitoring() {
  // Currently not implemented.
}

int portable_thermal_load() {
  // Currently not implemented.
  return 0;
}

void portable_start_system_load_advisory_monitoring() {
  // Currently not implemented.
}

int portable_system_load_advisory() {
  // Currently not implemented.
  return 0;
}

void portable_start_memory_pressure_monitoring() {
  // Currently not implemented.
  // https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
}

MemoryPressureLevel portable_memory_pressure() {
  // Currently not implemented.
  return MemoryPressureLevelNormal;
}

void portable_start_disk_space_monitoring() {
  // Currently not implemented.
}

void portable_start_cpu_speed_monitoring() {
  // Currently not implemented.
}

int portable_cpu_speed() {
  // Currently not implemented.
  return -1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_profiler_SystemNetworkStats_getNetIoCountersNative(
    JNIEnv *env, jclass clazz, jobject counters_map) {
  // Currently not implemented.
}

}  // namespace blaze_jni
