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

#include <errno.h>
#include <unistd.h>
#include <memory>
#include <string>

#include "src/tools/singlejar/input_jar.h"

#include "gtest/gtest.h"

static const char kJar[] = "jar.jar";

class InputJarBadJarTest : public testing::Test {
 protected:
  void SetUp() override {
    input_jar_.reset(new InputJar);
  }

  // Allocates a with given name and with given size.
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

  std::unique_ptr<InputJar> input_jar_;
};

TEST_F(InputJarBadJarTest, NotAJar) {
  ASSERT_EQ(0, chdir(getenv("TEST_TMPDIR")));
  AllocateFile(kJar, 1000);
  ASSERT_FALSE(input_jar_->Open(kJar));
}

// Check that an empty file does not cause trouble in MappedFile.
TEST_F(InputJarBadJarTest, EmptyFile) {
  ASSERT_EQ(0, chdir(getenv("TEST_TMPDIR")));
  AllocateFile(kJar, 0);
  ASSERT_FALSE(input_jar_->Open(kJar));
}
