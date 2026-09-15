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

#include <string>

#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "googletest/include/gtest/gtest.h"

static const char kJar[] = "jar.jar";

TEST(InputJarBadJarTest, NotAJar) {
  std::string out_path = singlejar_test_util::OutputFilePath(kJar);
  ASSERT_TRUE(singlejar_test_util::AllocateFile(out_path, 1000));
  InputJar input_jar;
  ASSERT_FALSE(input_jar.Open(out_path));
}

// Check that an empty file does not cause trouble in MappedFile.
TEST(InputJarBadJarTest, EmptyFile) {
  std::string out_path = singlejar_test_util::OutputFilePath(kJar);
  ASSERT_TRUE(singlejar_test_util::AllocateFile(out_path, 0));
  InputJar input_jar;
  ASSERT_FALSE(input_jar.Open(out_path));
}
