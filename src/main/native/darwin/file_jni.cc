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
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <sys/types.h>
#include <sys/xattr.h>

#include <string>

#include "src/main/native/unix_jni.h"

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

uint64_t StatEpochMilliseconds(const portable_stat_struct &statbuf,
                               StatTimes t) {
  switch (t) {
    case STAT_ATIME:
      return statbuf.st_atimespec.tv_sec * 1000L +
             statbuf.st_atimespec.tv_nsec / 1000000;
    case STAT_CTIME:
      return statbuf.st_ctimespec.tv_sec * 1000L +
             statbuf.st_ctimespec.tv_nsec / 1000000;
    case STAT_MTIME:
      return statbuf.st_mtimespec.tv_sec * 1000L +
             statbuf.st_mtimespec.tv_nsec / 1000000;
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

}  // namespace blaze_jni
