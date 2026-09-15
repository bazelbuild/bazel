// Copyright 2017 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_TEST_CPP_UTIL_TEST_UTIL_H_
#define BAZEL_SRC_TEST_CPP_UTIL_TEST_UTIL_H_

#include <stdio.h>

#if !defined(_WIN32) && !defined(__CYGWIN__)
#include <unistd.h>
#endif  // not _WIN32 and not __CYGWIN__

namespace blaze_util {

template <typename handle_type, handle_type null_value,
          int (*close_func)(handle_type)>
struct AutoFileHandle {
  AutoFileHandle(handle_type _fd = null_value) : fd(_fd) {}

  ~AutoFileHandle() { Close(); }

  int Close() {
    int result = 0;
    if (fd != null_value) {
      result = close_func(fd);
      fd = null_value;
    }
    return result;
  }

  bool IsOpen() { return fd != null_value; }

  operator handle_type() const { return fd; }

  AutoFileHandle& operator=(const handle_type& rhs) {
    Close();
    fd = rhs;
    return *this;
  }

  handle_type fd;
};

#if !defined(_WIN32) && !defined(__CYGWIN__)
typedef struct AutoFileHandle<int, -1, close> AutoFd;
#endif  // not _WIN32 and not __CYGWIN__

typedef struct AutoFileHandle<FILE*, nullptr, fclose> AutoFileStream;

}  // namespace blaze_util

#endif  // BAZEL_SRC_TEST_CPP_UTIL_TEST_UTIL_H_
