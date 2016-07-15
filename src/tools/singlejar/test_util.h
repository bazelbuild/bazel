#ifndef SRC_TOOLS_SINGLEJAR_TEST_UTIL_H_
#define SRC_TOOLS_SINGLEJAR_TEST_UTIL_H_
// Copyright 2016 The Bazel Authors. All rights reserved.
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

#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>

class TestUtil {
 public:
  // Allocate a file with given name and size. The contents is zeroes.
  static bool AllocateFile(const char *name, size_t size) {
    int fd = open(name, O_CREAT | O_RDWR | O_TRUNC, 0777);
    if (fd < 0) {
      perror(name);
      return false;
    }
    if (size) {
      if (ftruncate(fd, size) == 0) {
        return close(fd) == 0;
      } else {
        auto last_error = errno;
        close(fd);
        errno = last_error;
        return false;
      }
    } else {
      return close(fd) == 0;
    }
  }

  // List zip file contents.
  static void LsZip(const char *zip_name) {
#if !defined(__APPLE__)
    std::string command = (std::string("unzip -v ") + zip_name).c_str();
    system(command.c_str());
#endif
  }
};
#endif  //  SRC_TOOLS_SINGLEJAR_TEST_UTIL_H_
